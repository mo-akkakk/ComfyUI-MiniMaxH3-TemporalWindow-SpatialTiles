"""Minimax H3 Temporal Window & Spatial Tiles against ComfyUI's real H3 conditioning, model and samplers (CPU, tiny random DiT).

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
          tile_audio="generate_once", color_match=False):
    return pack.MiniMaxH3TemporalWindow.execute(model, frames > 0, max(frames, 22), overlap, short_final, min_final, True, tiles, tw, th, tov, True,
                                                tile_audio=tile_audio, color_match=color_match, clip=clip).result


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
    p, fz, band = hw.band_profile(20, 8)
    check("tile band: outer half frozen, inner half ramps, rest free",
          bool((p[:fz] == 0).all()) and fz == 4 and bool((p[fz:band] > 0).all() and (p[fz:band] < 1).all()) and bool((p[band:] == 1).all()))
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


def test_sequential_tiles(model):
    W, H, N = 640, 384, 56
    plate, ref = torch.rand(N, H, W, 3), torch.rand(1, H, W, 3)
    tw, th, tov = 384, 224, 96
    ys, xs = hw.plan_tiles(H // 16, th // 16, tov // 16), hw.plan_tiles(W // 16, tw // 16, tov // 16)
    tiles = [(yi, xi) for yi in range(len(ys)) for xi in range(len(xs))]
    for mode in ("generate_once", "keep_input"):
        m, c = patch(model, FakeClip(), frames=56, overlap=22, tiles=True, tw=tw, th=th, tov=tov, tile_audio=mode, color_match=True)
        pos, lat = conditioning(c, plate, ref, W, H, N, guide=False)
        v, a = lat["samples"].unbind()
        lat = {"samples": comfy.nested_tensor.NestedTensor((torch.randn_like(v), torch.randn_like(a)))}
        a_in = lat["samples"].unbind()[1]
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 4)[1:]
        out, pieces = sample_pieces(m, pos, lat, sigmas)
        ok = len(pieces) == len(tiles) and bool(torch.isfinite(out[0]).all())
        ok_frozen = ok_audio = True
        for k, (yi, xi) in enumerate(tiles):
            p = pieces[k]
            (y0, y1), (x0, x1) = ys[yi], xs[xi]
            top, left = hw.tile_bands(ys, xs, yi, xi)
            if mode == "generate_once" and k == 0:
                ok_audio &= p["mask"] is None or bool((p["mask"][1] > 0).all())
            else:
                ok_audio &= p["mask"] is not None and bool((p["mask"][1] == 0).all())
            if k == 0:
                continue
            _, fy, _ = hw.band_profile(y1 - y0, top)
            _, fx, _ = hw.band_profile(x1 - x0, left)
            mv = p["mask"][0]
            ok_frozen &= bool((mv[..., :fy, :] == 0).all()) and bool((mv[..., :, :fx] == 0).all())
            # the frozen band comes back exactly as the finished neighbour left it
            if fy:
                ok_frozen &= torch.allclose(p["out"][0][..., :fy, :], out[0][..., y0:y0 + fy, x0:x1], atol=1e-5) or yi + 1 < len(ys)
                ok_frozen &= torch.allclose(p["out"][0][..., :fy, :], p["lat"][0][..., :fy, :], atol=1e-5)
            if fx:
                ok_frozen &= torch.allclose(p["out"][0][..., :, :fx], p["lat"][0][..., :, :fx], atol=1e-5)
        if mode == "generate_once":
            ok_audio &= torch.allclose(out[1], pieces[0]["out"][1], atol=1e-5)
        else:
            ok_audio &= torch.allclose(out[1], a_in, atol=1e-4)
        check("tiles ({}): {} tiles one after another, finished output".format(mode, len(tiles)), bool(ok))
        check("tiles ({}): each tile's band with finished neighbours is prefilled and frozen".format(mode), bool(ok_frozen))
        check("tiles ({}): soundtrack {}".format(mode, "generated by the first tile and held by the rest" if mode == "generate_once" else "kept from the input in every tile"),
              bool(ok_audio))


if __name__ == "__main__":
    test_plans()
    test_short_final_window_plan()
    test_guide_helpers()
    with tempfile.TemporaryDirectory() as d:
        model = tiny_model(os.path.join(d, "tiny_h3.safetensors"))
        test_one_window_is_native(model)
        test_switch_off_is_native(model)
        test_short_final_window_sampling(model)
        test_windows(model)
        test_ksampler(model)
        test_stage2_tiles(model)
        test_sequential_tiles(model)
    print("\n{} failure(s)".format(len(FAILS)))
    sys.exit(1 if FAILS else 0)
