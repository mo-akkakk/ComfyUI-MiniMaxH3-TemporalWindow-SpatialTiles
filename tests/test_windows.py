"""Minimax H3 V2V Temporal Window & Spatial Tiles against ComfyUI's real H3 conditioning, model and samplers (CPU, tiny random DiT).

    python custom_nodes/ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles/tests/test_windows.py   (from the ComfyUI root)
"""

import copy
import importlib
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
import comfy.cli_args  # noqa: E402

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy_extras.nodes_minimax_h3 as h3  # noqa: E402
import nodes  # noqa: E402
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402

PACK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(PACK))
pack = importlib.import_module(os.path.basename(PACK))
hw = importlib.import_module(os.path.basename(PACK) + ".h3_windows")

TEXT_DIM = 64
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def tiny_model(path):
    torch.manual_seed(0)
    m = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                       attention_head_dim=128, ffn_hidden_size=512, text_dim=TEXT_DIM,
                       timestep_input_dim=256, time_embed_hidden_size=256, time_embed_dim=128,
                       operations=comfy.ops.disable_weight_init)
    sd = {}
    for k, v in m.state_dict().items():
        if k.endswith("rope.inv_freq"):
            sd[k] = 1.0 / (10000.0 ** (torch.arange(16, dtype=torch.float32) / 16.0))
        elif "norm" in k and k.endswith("weight"):
            sd[k] = torch.ones_like(v, dtype=torch.float32)
        else:
            sd[k] = torch.randn(v.shape, dtype=torch.float32) * 0.05
    safetensors.torch.save_file(sd, path)
    return comfy.sd.load_diffusion_model(path)


class FakeClip:
    """Deterministic Qwen stand-in that records what each encode is shown."""

    def __init__(self, log=None):
        self.log = log if log is not None else []

    def clone(self):
        return FakeClip(self.log)

    def tokenize(self, text, return_word_ids=False, minimax_ref_items=None, images=None):
        return {"items": minimax_ref_items or []}

    def encode_from_tokens_scheduled(self, tokens):
        items = tokens["items"]
        self.log.append(items)
        key = sum(float(it["data"].float().mean()) * (i + 1) for i, it in enumerate(items) if "data" in it)
        g = torch.Generator().manual_seed(int(abs(key) * 1e6) % (2 ** 31))
        n = 8 + sum(1 + it["data"].shape[0] if "data" in it else 1 for it in items)
        return [[torch.randn(1, n, TEXT_DIM, generator=g), {"minimax_token_tags": torch.ones(1, n, dtype=torch.long)}]]


class FakeVAE:
    def encode(self, frames):
        t = 1 if frames.shape[0] == 1 else h3.video_latent_t(frames.shape[0])
        g = torch.Generator().manual_seed(int(float(frames.float().mean()) * 1e6) % (2 ** 31))
        return torch.randn(1, 24, t, frames.shape[1] // 16, frames.shape[2] // 16, generator=g)


def conditioning(clip, plate, ref, W, H, N, guide=True):
    pos, lat = h3.MiniMaxH3ReferenceToVideo.execute(clip, "p", W, H, N, "match", FakeVAE(), None,
                                                     {"ref_image_1": ref}, {"ref_video_1": plate}, None, None).result
    if guide:
        pos = h3.MiniMaxH3AddGuide.execute(pos, lat, 0, vae=FakeVAE(), image=ref).result[0]
    return pos, lat


def run(model, pos, lat, sigmas, seed=3):
    g = Guider_Basic(model)
    g.set_conds(pos)
    return SamplerCustomAdvanced.execute(Noise_RandomNoise(seed), g, comfy.samplers.sampler_object("euler"), sigmas, lat).result


def patch(model, clip, frames=124, overlap=22, tiles=False, tw=960, th=544, tov=256, short_final=False, min_final=22,
          tile_audio="generate_once", color_match=False, tile_carry="own_tile", plate=None, vae=None, plate_scale=1.0):
    return pack.MiniMaxH3TemporalWindow.execute(model, frames > 0, max(frames, 22), overlap, short_final, min_final, True, tiles, tw, th, tov, True,
                                                tile_audio=tile_audio, color_match=color_match,
                                                tile_carry=tile_carry, plate_scale=plate_scale, clip=clip, plate=plate, vae=vae).result


def test_plans():
    for frames in (124, 209, 362, 719):
        T = h3.temporal_shape(frames)[1]
        w = hw.plan_windows(T, 124, 22)
        ok = w[0][0] == 0 and w[-1][0] + w[-1][1] == T
        ok &= all(s % 5 == 0 and (n - 2) % 5 == 0 for s, n in w)
        ok &= all(w[i][0] + w[i][1] - w[i + 1][0] >= hw.tokens_for_frames(22) for i in range(len(w) - 1))
        check("window plan {} frames -> {}".format(frames, [(hw.frames_for_tokens(s), hw.frames_for_tokens(n)) for s, n in w]), ok)


def test_switch_off_is_native(model):
    W, H, N = 384, 256, 209
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 1)
    pos, lat = conditioning(FakeClip(), plate, ref, W, H, N)
    native = run(model, pos, lat, sigmas)[0]["samples"].unbind()
    m, c = patch(model, FakeClip(), frames=0)
    pos2, lat2 = conditioning(c, plate, ref, W, H, N)
    mine = run(m, pos2, lat2, sigmas)[0]["samples"].unbind()
    check("temporal_windows off: whole length in one native pass", torch.equal(native[0], mine[0]) and torch.equal(native[1], mine[1]))


def test_short_final_window_plan():
    T = h3.temporal_shape(362)[1]
    F = hw.frames_for_tokens
    full = hw.plan_windows(T, 243, 22)
    short = hw.plan_windows(T, 243, 22, 22)
    check("default: final window pulled back to full length {}".format([(F(s), F(n)) for s, n in full]), F(full[-1][1]) == 243)
    check("short final window keeps its natural length {}".format([(F(s), F(n)) for s, n in short]),
          F(short[-1][1]) == 141 and short[-1][0] + short[-1][1] == T)
    T2 = h3.temporal_shape(719)[1]
    w = hw.plan_windows(T2, 192, 22, 124)
    check("short tail below the minimum is pulled back only to the minimum {}".format([(F(s), F(n)) for s, n in w]),
          F(w[-1][1]) == 124 and w[-1][0] + w[-1][1] == T2)
    ok = True
    for frames in (124, 209, 362, 400, 719):
        Tn = h3.temporal_shape(frames)[1]
        for mn in (5, 22, 39, 56):
            ws = hw.plan_windows(Tn, 124, 22, mn)
            ok &= ws[0][0] == 0 and ws[-1][0] + ws[-1][1] == Tn and all(s % 5 == 0 and (n - 2) % 5 == 0 for s, n in ws)
            ok &= all(ws[i][0] + ws[i][1] > ws[i + 1][0] for i in range(len(ws) - 1))
            ok &= F(ws[-1][1]) >= min(mn, 124)
    check("every plan with a short final window is valid, overlapping and covers the shot", bool(ok))


def test_short_final_window_sampling(model):
    W, H, N = 384, 256, 260
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    m, c = patch(model, FakeClip(), frames=124, overlap=22, short_final=True, min_final=5)
    pos, lat = conditioning(c, plate, ref, W, H, N)
    T = lat["samples"].unbind()[0].shape[2]
    wins = hw.plan_windows(T, 124, 22, 5)
    lengths = []
    orig = comfy.samplers.CFGGuider.inner_sample

    def rec(self, *a, **k):
        lengths.append(k["latent_shapes"][0][2])
        return orig(self, *a, **k)
    comfy.samplers.CFGGuider.inner_sample = rec
    try:
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 1)
        out = run(m, pos, lat, sigmas)[0]["samples"].unbind()
    finally:
        comfy.samplers.CFGGuider.inner_sample = orig
    check("sampling with a short final window ({} frames) covers the shot".format(hw.frames_for_tokens(wins[-1][1])),
          lengths == [n for _, n in wins] and wins[-1][1] < hw.tokens_for_frames(124)
          and out[0].shape == lat["samples"].unbind()[0].shape and bool(torch.isfinite(out[0]).all()))


def test_one_window_is_native(model):
    W, H, N = 384, 256, 56
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 2)
    pos, lat = conditioning(FakeClip(), plate, ref, W, H, N)
    native = run(model, pos, lat, sigmas)[0]["samples"].unbind()
    m, c = patch(model, FakeClip())
    pos2, lat2 = conditioning(c, plate, ref, W, H, N)
    mine = run(m, pos2, lat2, sigmas)[0]["samples"].unbind()
    check("one window, no tiles: bit-identical to native", torch.equal(native[0], mine[0]) and torch.equal(native[1], mine[1]))


def test_windows(model):
    W, H, N = 384, 256, 209
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    log = []
    m, c = patch(model, FakeClip(log), frames=56, overlap=22)
    pos, lat = conditioning(c, plate, ref, W, H, N)
    T = lat["samples"].unbind()[0].shape[2]
    wins = hw.plan_windows(T, 56, 22)

    pieces = []
    orig = comfy.samplers.CFGGuider.inner_sample

    def rec(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        pos_c = self.conds["positive"][0]
        pieces.append({"shapes": latent_shapes, "refs": pos_c.get("minimax_refs"),
                       "kfs": pos_c.get("minimax_keyframes") or [], "ctx_len": pos_c["cross_attn"].shape[1]})
        out = orig(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        pieces[-1]["out"] = comfy.utils.unpack_latents(out.cpu(), latent_shapes)
        return out
    comfy.samplers.CFGGuider.inner_sample = rec
    try:
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 2)
        out, denoised = run(m, pos, lat, sigmas)
    finally:
        comfy.samplers.CFGGuider.inner_sample = orig
    v, a = out["samples"].unbind()
    check("{} windows sampled one after another, output covers all {} frames".format(len(wins), N),
          len(pieces) == len(wins) and v.shape == lat["samples"].unbind()[0].shape and bool(torch.isfinite(v).all()))
    check("denoised_output is the full latent", tuple(denoised["samples"].unbind()[0].shape) == tuple(v.shape))

    full_ref = pos[0][1]["minimax_refs"][1]["latent"]
    ok_ref = ok_guide = ok_carry = True
    for k, (t0, tn) in enumerate(wins):
        p = pieces[k]
        ok_ref &= p["shapes"][0][2] == tn and torch.equal(p["refs"][1]["latent"], full_ref[:, :, t0:t0 + tn])
        user = [kf for kf in p["kfs"] if kf["latent"].shape[2] == 1]
        ok_guide &= (len(user) == 1) if k == 0 else (len(user) == 0)
        if k > 0:
            carry = [kf for kf in p["kfs"] if kf["latent"].shape[2] > 1]
            p0, pn = wins[k - 1]
            ok_carry &= len(carry) == 1 and carry[0]["resolved_frame_index"] == 0 and carry[0]["latent"].shape[2] == p0 + pn - t0
            ok_carry &= carry[0].get("audio_latent") is not None
            head_end = max([s + n for s, n in wins[:k - 1]], default=0)
            if t0 >= head_end:  # window k-1's tail is its own, un-crossfaded output
                prev = pieces[k - 1]["out"][0][:, :, t0 - p0:t0 - p0 + carry[0]["latent"].shape[2]]
                ok_carry &= torch.allclose(carry[0]["latent"], prev, atol=1e-5)
    check("each window's <Video 1> latents are its slice of the plate", bool(ok_ref))
    check("the frame-0 guide only lands in window 1", bool(ok_guide))
    check("window k starts from window k-1's finished overlap (video + audio)", bool(ok_carry))

    # per-window prompt: Qwen sees only that window of <Video 1>, timestamps from 0
    window_logs = log[1:]
    full_vid = next(it for it in log[0] if it["type"] == "video")
    ok = len(window_logs) == len(wins)
    for items, (t0, tn) in zip(window_logs, wins):
        vid = next(it for it in items if it["type"] == "video")
        f0, fn = hw.frames_for_tokens(t0), hw.frames_for_tokens(tn)
        keep = [i for i, ts in enumerate(full_vid["timestamps"]) if f0 / 24 <= ts < (f0 + fn) / 24]
        ok &= torch.equal(vid["data"], full_vid["data"][keep])
        ok &= vid["timestamps"] == [full_vid["timestamps"][i] - full_vid["timestamps"][keep[0]] for i in keep]
    check("prompt re-encoded per window: Qwen sees only the window's frames of <Video 1>, timestamps from 0", bool(ok),
          "{} encodes".format(len(window_logs)))
    check("each window ran with its own prompt encoding", len({p["ctx_len"] for p in pieces}) >= 1 and len(pieces) == len(window_logs))


def test_ksampler(model):
    W, H, N = 384, 256, 124
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    m, c = patch(model, FakeClip(), frames=56, overlap=22)
    pos, lat = conditioning(c, plate, ref, W, H, N)
    out = nodes.common_ksampler(m, 1, 2, 1.0, "euler", "simple", pos, pos, lat)[0]["samples"].unbind()
    check("works through KSampler too", out[0].shape == lat["samples"].unbind()[0].shape and bool(torch.isfinite(out[0]).all()))


def test_stage2_tiles(model):
    W, H, N = 640, 384, 124
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    log = []
    m, c = patch(model, FakeClip(log), frames=56, overlap=22, tiles=True, tw=384, th=224, tov=96)
    pos, lat = conditioning(c, plate, ref, W, H, N, guide=False)
    v, a = lat["samples"].unbind()
    lat = {"samples": comfy.nested_tensor.NestedTensor((torch.randn_like(v), torch.randn_like(a)))}
    T = v.shape[2]
    wins = hw.plan_windows(T, 56, 22)
    ys, xs = hw.plan_tiles(H // 16, 224 // 16, 96 // 16), hw.plan_tiles(W // 16, 384 // 16, 96 // 16)
    sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 4)[2:]
    out = run(m, pos, lat, sigmas)[0]["samples"].unbind()
    n_pieces = len(wins) * len(ys) * len(xs)
    check("stage 2: {} windows x {} tiles, partial sigmas, full-size output".format(len(wins), len(ys) * len(xs)),
          out[0].shape == v.shape and bool(torch.isfinite(out[0]).all()))
    crops = [next(it for it in items if it["type"] == "video")["data"].shape[1:3] for items in log[1:]]
    check("per-tile prompt sees the tile's crop of <Video 1>", len(crops) == n_pieces and all(h < H and w < W for h, w in crops), str(sorted(set(crops))))


def test_guide_helpers():
    F = hw.frames_for_tokens
    # a 22-frame guide clip (7 latents) at frame 39 crosses the end of a 56-frame window
    clip = torch.randn(1, 24, 7, 4, 6)
    audio = torch.randn(1, 32, 2, hw.audio_ticks(22))
    kfs = [{"resolved_frame_index": 39, "latent": clip, "audio_latent": audio, "strength": 1.0}]
    out = hw.window_keyframes(kfs, 0, hw.tokens_for_frames(56), set())
    vid = [k for k in out if k.get("latent") is not None]
    aud = [k for k in out if k.get("latent") is None]
    ok = len(vid) == 1 and vid[0]["resolved_frame_index"] == 39 and torch.equal(vid[0]["latent"], clip[:, :, :5]) and vid[0].get("strength") == 1.0
    ok &= len(aud) == 1 and aud[0]["audio_latent"].shape[-1] >= 1
    ok &= aud[0]["resolved_frame_index"] * hw.FRAME_RESCALE + aud[0]["audio_latent"].shape[-1] <= hw.audio_ticks(56) + 1
    check("a guide clip crossing the window edge keeps its 17-frame chunk inside (video) and its audio inside", bool(ok))
    # next window starts at frame 51 (latent 15): only the clip's second chunk (frames 56-60) is inside
    out = hw.window_keyframes(kfs, 15, 15 + hw.tokens_for_frames(56), set())
    vid = [k for k in out if k.get("latent") is not None]
    ok = len(vid) == 1 and vid[0]["resolved_frame_index"] == 56 - F(15) and torch.equal(vid[0]["latent"], clip[:, :, 5:])
    check("the next window keeps the clip's later chunk, re-timed", bool(ok))
    r = hw.resize_keyframes([{"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, 8, 12)}, {"resolved_frame_index": 0, "audio_latent": audio}], 16, 24, set())
    check("guide latents at another resolution are resized to the canvas", tuple(r[0]["latent"].shape) == (1, 24, 1, 16, 24) and "latent" not in r[1])
    a, b = torch.randn(1, 24, 2, 8, 8), torch.randn(1, 24, 2, 8, 8)
    dc = hw.dc_offset([(a + 1.0, a)])
    check("colour match offset is clamped", dc is not None and bool(torch.allclose(dc, torch.full((24,), hw.COLOR_CLAMP))))


def sample_pieces(m, pos, lat, sigmas):
    pieces = []
    orig = comfy.samplers.CFGGuider.inner_sample

    def rec(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        piece = {"shapes": latent_shapes, "lat": comfy.utils.unpack_latents(latent_image.cpu(), latent_shapes),
                 "mask": comfy.utils.unpack_latents(denoise_mask.cpu(), latent_shapes) if denoise_mask is not None else None}
        out = orig(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        piece["out"] = comfy.utils.unpack_latents(out.cpu(), latent_shapes)
        pieces.append(piece)
        return out
    comfy.samplers.CFGGuider.inner_sample = rec
    try:
        out = run(m, pos, lat, sigmas)[0]["samples"].unbind()
    finally:
        comfy.samplers.CFGGuider.inner_sample = orig
    return out, pieces


def test_tile_audio(model):
    W, H, N = 640, 384, 56
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    tw, th, tov = 384, 224, 96
    ys, xs = hw.plan_tiles(H // 16, th // 16, tov // 16), hw.plan_tiles(W // 16, tw // 16, tov // 16)
    for mode in ("generate_once", "keep_input"):
        m, c = patch(model, FakeClip(), frames=56, overlap=22, tiles=True, tw=tw, th=th, tov=tov, tile_audio=mode, color_match=True)
        pos, lat = conditioning(c, plate, ref, W, H, N, guide=False)
        v, a = lat["samples"].unbind()
        lat = {"samples": comfy.nested_tensor.NestedTensor((torch.randn_like(v), torch.randn_like(a)))}
        a_in = lat["samples"].unbind()[1]
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 4)[1:]
        out, pieces = sample_pieces(m, pos, lat, sigmas)
        ok = len(pieces) == len(ys) * len(xs) and bool(torch.isfinite(out[0]).all()) and out[0].shape == v.shape
        ok &= all(p["mask"] is None for p in pieces)
        check("tiles ({}): independent tiles, no masks, finished output".format(mode), bool(ok))
        if mode == "generate_once":
            check("tiles: soundtrack is the first tile's", torch.allclose(out[1], pieces[0]["out"][1], atol=1e-5))
        else:
            check("tiles: keep_input outputs the incoming audio", torch.allclose(out[1], a_in, atol=1e-4))


def test_tile_carry(model):
    W, H, N = 640, 384, 124
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    tw, th, tov = 384, 224, 96
    ys, xs = hw.plan_tiles(H // 16, th // 16, tov // 16), hw.plan_tiles(W // 16, tw // 16, tov // 16)
    tiles = [(yi, xi) for yi in range(len(ys)) for xi in range(len(xs))]
    for mode in ("own_tile", "blended"):
        m, c = patch(model, FakeClip(), frames=73, overlap=22, tiles=True, tw=tw, th=th, tov=tov, tile_carry=mode)
        pos, lat = conditioning(c, plate, ref, W, H, N, guide=False)
        carries = []
        orig_pc = hw.piece_conds

        def spy(conds, k0, k1, ys_, xs_, *a, **kw):
            if a[3] is not None:
                carries.append((k0, ys_, xs_, a[3]["latent"].clone()))
            return orig_pc(conds, k0, k1, ys_, xs_, *a, **kw)
        hw.piece_conds = spy
        try:
            sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 3)
            out, pieces = sample_pieces(m, pos, lat, sigmas)
        finally:
            hw.piece_conds = orig_pc
        windows = hw.plan_windows(lat["samples"].unbind()[0].shape[2], 73, 22)
        n = len(tiles)
        ok = len(windows) >= 2 and len(carries) == n * (len(windows) - 1) and len(pieces) == n * len(windows)
        own_match = blended_diff = True
        for j, (k0, ys_, xs_, c_lat) in enumerate(carries):
            ti = j % n
            w_prev = j // n
            p0 = windows[w_prev][0]
            raw = pieces[w_prev * n + ti]["out"][0][:, :, k0 - p0:k0 - p0 + c_lat.shape[2]].float()
            own_match &= torch.allclose(c_lat, raw, atol=1e-5)
        if mode == "own_tile":
            check("tile_carry own_tile: each tile continues from its own unblended frames ({} windows x {} tiles)".format(len(windows), n), bool(ok and own_match))
        else:
            check("tile_carry blended: tiles continue from the blended picture (1.0.6 behaviour)", bool(ok and not own_match))


def test_exact_crop():
    lat_h, lat_w = 96, 168           # 2688x1536 canvas
    rh, rw = 48, 84                  # 1344x768 plate
    yy = (torch.arange(rh, dtype=torch.float32) + 0.5) / rh
    xx = (torch.arange(rw, dtype=torch.float32) + 0.5) / rw
    ramp = (yy.view(-1, 1) * 3 + xx.view(1, -1) * 5).expand(1, 24, 3, rh, rw).contiguous()
    worst_new = worst_old = 0.0
    for ys in hw.plan_tiles(lat_h, 34, 16):
        for xs in hw.plan_tiles(lat_w, 60, 16):
            out = hw.exact_crop(ramp, ys, xs, lat_h, lat_w)
            oh, ow = out.shape[-2:]
            ty = (ys[0] + (torch.arange(oh) + 0.5) * (ys[1] - ys[0]) / oh) / lat_h
            tx = (xs[0] + (torch.arange(ow) + 0.5) * (xs[1] - xs[0]) / ow) / lat_w
            want = ty.view(-1, 1) * 3 + tx.view(1, -1) * 5
            worst_new = max(worst_new, float((out[0, 0, 0] - want).abs()[1:-1, 1:-1].max()))
            ry0, ry1, rx0, rx1 = hw.crop_box(ys, xs, lat_h, lat_w, rh, rw)
            old = ramp[0, 0, 0, ry0:ry1, rx0:rx1]
            ty = (ys[0] + (torch.arange(ry1 - ry0) + 0.5) * (ys[1] - ys[0]) / (ry1 - ry0)) / lat_h
            tx = (xs[0] + (torch.arange(rx1 - rx0) + 0.5) * (xs[1] - xs[0]) / (rx1 - rx0)) / lat_w
            worst_old = max(worst_old, float((old - (ty.view(-1, 1) * 3 + tx.view(1, -1) * 5)).abs().max()))
    check("exact tile crop on a smaller plate grid lands where the tile is (error {:.3f}, old rounded crop {:.3f})".format(worst_new, worst_old),
          worst_new < 0.02 and worst_old > 10 * worst_new)
    same = torch.randn(1, 24, 3, 48, 84)
    check("exact tile crop on the same grid is a plain slice (stage 1 unchanged)",
          torch.equal(hw.exact_crop(same, (14, 48), (24, 84), 48, 84), same[..., 14:48, 24:84]))


def test_full_res_plate(model):
    W, H, N = 640, 384, 124
    torch.manual_seed(7)
    plate = torch.rand(N, H, W, 3)
    plate2x = torch.nn.functional.interpolate(plate.movedim(-1, 1), scale_factor=2, mode="bilinear").movedim(1, -1)
    ref = torch.rand(1, H, W, 3)
    tw, th, tov = 384, 224, 96
    ys, xs = hw.plan_tiles(H // 16, th // 16, tov // 16), hw.plan_tiles(W // 16, tw // 16, tov // 16)
    m, c = patch(model, FakeClip(), frames=73, overlap=22, tiles=True, tw=tw, th=th, tov=tov, plate=plate2x, vae=FakeVAE())
    pos, lat = conditioning(c, plate, ref, W, H, N, guide=False)
    seen = []
    orig_pc = hw.piece_conds

    def spy(conds, k0, k1, ys_, xs_, *a, **kw):
        out = orig_pc(conds, k0, k1, ys_, xs_, *a, **kw)
        vid = [b for b in out["positive"][0]["minimax_refs"] if b["kind"] in ("video", "video_audio")][0]
        seen.append((k0, k1, ys_, xs_, vid))
        return out
    hw.piece_conds = spy
    try:
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 3)[1:]
        out, pieces = sample_pieces(m, pos, lat, sigmas)
    finally:
        hw.piece_conds = orig_pc
    ok = len(pieces) > 0 and bool(torch.isfinite(out[0]).all())
    ok_shape = all(v["latent"].shape[-2:] == (y[1] - y[0], x[1] - x[0]) and v["latent"].shape[2] == v["latent_t"]
                   for k0, k1, y, x, v in seen)
    k0, k1, y, x, v = seen[0]
    f0 = hw.frames_for_tokens(k0)
    crop = plate2x[f0:f0 + hw.frames_for_tokens(v["latent_t"]), y[0] * 32:y[1] * 32, x[0] * 32:x[1] * 32]
    pix = comfy.utils.common_upscale(crop.movedim(-1, 1), (x[1] - x[0]) * 16, (y[1] - y[0]) * 16, "lanczos", "disabled").movedim(1, -1)
    ok_px = torch.equal(v["latent"], FakeVAE().encode(pix))
    check("full-res plate: every tile's <Video 1> is encoded from its exact crop of the plate ({} pieces sampled)".format(len(pieces)),
          bool(ok and ok_shape and ok_px))

    m, c = patch(model, FakeClip(), frames=73, overlap=22, tiles=True, tw=tw, th=th, tov=tov, plate=plate2x, vae=FakeVAE(), plate_scale=0.5)
    pos, lat = conditioning(c, plate, ref, W, H, N, guide=False)
    seen.clear()
    hw.piece_conds = spy
    try:
        out, pieces = sample_pieces(m, pos, lat, sigmas)
    finally:
        hw.piece_conds = orig_pc
    ok = bool(torch.isfinite(out[0]).all()) and len(seen) > 0
    ok &= all(abs(v["latent"].shape[-2] - (y[1] - y[0]) / 2) <= 1 and abs(v["latent"].shape[-1] - (x[1] - x[0]) / 2) <= 1
              for k0, k1, y, x, v in seen)
    check("plate_scale 0.5: each tile's plate piece is half the tile size (a quarter of the reference tokens)", ok)


class GenClip:
    """Text-aware Qwen stand-in for the Long Generation node: records the text and images of every encode."""

    def __init__(self, log=None):
        self.log = log if log is not None else []

    def clone(self):
        return GenClip(self.log)

    def tokenize(self, text, return_word_ids=False, minimax_ref_items=None, images=None):
        return {"text": text, "images": list(images or []), "items": minimax_ref_items or []}

    def encode_from_tokens_scheduled(self, tokens):
        self.log.append((tokens["text"], tokens["images"], tokens["items"]))
        key = sum(ord(ch) for ch in tokens["text"]) + sum(float(im.mean()) for im in tokens["images"])
        g = torch.Generator().manual_seed(int(key * 1000) % (2 ** 31))
        n = 8 + 2 * len(tokens["images"])
        return [[torch.randn(1, n, TEXT_DIM, generator=g), {"minimax_token_tags": torch.ones(1, n, dtype=torch.long)}]]


def test_schedule_parsing():
    e = hw.parse_schedule("0s: a man walks in\n6s: he sits down\nslowly\n# note\n[12.5s-15s] he stands up")
    ok = e == [(0.0, None, "a man walks in"), (6.0, None, "he sits down\nslowly"), (12.5, 15.0, "he stands up")]
    ok &= hw.parse_schedule("just one prompt") == [(0.0, None, "just one prompt")] and hw.parse_schedule("  ") == []
    lines = hw.window_timeline(e, 4.0, 9.2)
    ok &= lines == ["[0s-2s] a man walks in", "[2s-5.2s] he sits down\nslowly"]
    ok &= hw.window_timeline(e, 12.0, 17.2) == ["[0s-0.5s] he sits down\nslowly", "[0.5s-3s] he stands up"]
    # an action starting in the window's last second waits for the next window
    ok &= hw.window_timeline(e, 0.0, 6.5) == ["[0s-6.5s] a man walks in"]
    ok &= hw.window_timeline(e, 4.9, 10.1) == ["[0s-1.1s] a man walks in", "[1.1s-5.2s] he sits down\nslowly"]
    ok &= hw.window_timeline(e, 5.8, 11.0) == ["[0s-5.2s] he sits down\nslowly"]
    scene, tl = hw.split_timeline("Snowy street look.\n\nTimeline:\n[0s-1s] walks\n[1s-2.5s] stops\n\nAudio: wind.")
    ok &= scene == "Snowy street look.\n\nAudio: wind." and tl == [(0.0, 1.0, "walks"), (1.0, 2.5, "stops")]
    check("prompt schedule: '6s:' and '[6s-10s]' lines, re-timed per window, no action announced in a window's last second; "
          "a Timeline in the prompt is found", ok)


def test_long_generation(model):
    W, H, N = 384, 256, 175
    first, last = torch.full((1, H, W, 3), 0.2), torch.full((1, H, W, 3), 0.8)
    sched = "0s: walks in\n2s: sits down\n4.5s: looks out"
    entries = hw.parse_schedule(sched)
    for policy in ("own_window", "all_windows"):
        log = []
        m, c = pack.MiniMaxH3LongGeneration.execute(model, sched, True, 56, 22, False, 22, True, policy, clip=GenClip(log)).result
        pos, lat = h3.MiniMaxH3ImageToVideo.execute(c, FakeVAE(), "A snowy street at dusk.", W, H, N, first_frame=first, last_frame=last).result[:2]
        n0 = len(log)
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 2)
        out = run(m, pos, lat, sigmas)[0]["samples"].unbind()
        lat_t = lat["samples"].unbind()[0].shape[2]
        windows = hw.plan_windows(lat_t, 56, 22)
        enc = log[n0:]
        total = hw.frames_for_tokens(lat_t)
        ok = len(windows) >= 3 and len(enc) == len(windows) and bool(torch.isfinite(out[0]).all())
        for k, (t0, tn) in enumerate(windows):
            f0, f1 = hw.frames_for_tokens(t0), hw.frames_for_tokens(t0 + tn)
            text, imgs, _ = enc[k]
            want = hw.window_prompt("A snowy street at dusk.", hw.window_timeline(entries, f0 / 24, f1 / 24), True)
            ok &= text == want and text.startswith("A snowy street at dusk.\n\nTimeline:\n[0s-") and text.endswith("One continuous shot, no cuts.")
            if policy == "own_window":
                want_imgs = [v for v, fi in ((0.2, 0), (0.8, total - 1)) if f0 <= fi < f1]
            else:
                want_imgs = [0.2, 0.8]
            ok &= [round(float(im.mean()), 1) for im in imgs] == want_imgs
        check("long generation ({}): {} windows each get the scene + their own re-timed timeline; first/last images {}".format(
            policy, len(windows), "only in their own window" if policy == "own_window" else "in every window"), bool(ok))
        if policy == "own_window":
            print("    e.g. window 2 prompt:", repr(enc[1][0]))
    log = []
    m, c = pack.MiniMaxH3LongGeneration.execute(model, "", False, 56, 22, False, 22, True, "own_window", clip=GenClip(log)).result
    prompt = "Look: snowy dusk.\nTimeline:\n[0s-3s] walks\n[3s-7.3s] stops"
    pos, lat = h3.MiniMaxH3ImageToVideo.execute(c, FakeVAE(), prompt, W, H, N, first_frame=first).result[:2]
    n0 = len(log)
    run(m, pos, lat, comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 2))
    enc = log[n0:]
    windows = hw.plan_windows(lat["samples"].unbind()[0].shape[2], 56, 22)
    ok = len(enc) == len(windows) and len(enc[0][1]) == 1 and all(len(e[1]) == 0 for e in enc[1:])
    f0 = hw.frames_for_tokens(windows[1][0]) / 24
    ok &= enc[1][0] == "Look: snowy dusk.\n\nTimeline:\n" + "\n".join(hw.window_timeline(hw.split_timeline(prompt)[1], f0, f0 + 56 / 24))
    check("long generation without a schedule: a Timeline in the prompt is re-timed per window; the first image only in window 1", ok)
    log = []
    m, c = pack.MiniMaxH3LongGeneration.execute(model, "", False, 56, 22, False, 22, True, "own_window", clip=GenClip(log)).result
    pos, lat = h3.MiniMaxH3ImageToVideo.execute(c, FakeVAE(), "plain prompt", W, H, N).result[:2]
    n0 = len(log)
    run(m, pos, lat, comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 2))
    check("long generation, plain prompt, continuous_shot off: every window uses the prompt unchanged",
          len(log) > n0 and all(e[0] == "plain prompt" for e in log[n0:]))


if __name__ == "__main__":


    test_plans()
    test_short_final_window_plan()
    test_guide_helpers()
    test_exact_crop()
    test_schedule_parsing()
    with tempfile.TemporaryDirectory() as d:
        model = tiny_model(os.path.join(d, "tiny_h3.safetensors"))
        test_one_window_is_native(model)
        test_switch_off_is_native(model)
        test_short_final_window_sampling(model)
        test_windows(model)
        test_ksampler(model)
        test_stage2_tiles(model)
        test_tile_audio(model)
        test_tile_carry(model)
        test_full_res_plate(model)
        test_long_generation(model)
    print("\n{} failure(s)".format(len(FAILS)))
    sys.exit(1 if FAILS else 0)
