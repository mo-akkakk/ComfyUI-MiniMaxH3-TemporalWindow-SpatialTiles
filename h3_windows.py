"""Minimax H3 Temporal Window & Spatial Tiles (mo-akkakk): temporal windows (stage 1 + 2) and spatial tiles (stage 2), as one model patch.

The patch wraps the guider's sampling run (OUTER_SAMPLE), so it works with KSampler,
SamplerCustomAdvanced and any guider, and every piece is an ordinary native H3 run:

  * windows are sampled one after another over the connected sigmas;
  * window k gets window k-1's finished frames over their overlap (video + audio)
    as a frame-0 guide clip, the latent form of "Add Guide with the previous batch's
    last frames", and the overlap is crossfaded ease-in-out;
  * each window's conditioning is the native one, re-timed to the window: reference
    videos sliced to the window, frame guides moved to window time (and dropped from
    windows that don't contain them);
  * with the CLIP routed through the node, the prompt is re-encoded per window, so Qwen
    sees only that window of <Video N> with timestamps from 0 (like a manual batch),
    and per tile with the matching crop;
  * stage 2 only: tiles inside a window are sampled one by one and feather-blended.
"""

import logging
import math

import torch

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE

WRAPPER_KEY = "minimax_h3_windows"
TEXT_SOURCE = "minimax_h3_text_source"
CLIP_TOKENS = 5
FPS = 24


def frames_for_tokens(t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(t))


def tokens_for_frames(frames):
    """Latent length of the largest valid H3 clip (17k + 5 frames) that fits in `frames`."""
    return 2 + 5 * max(0, (frames - 5) // 17)


def audio_ticks(frames):
    return round(frames * FRAME_RESCALE)


def plan_windows(latent_t, window_frames, overlap_frames):
    """[(start, length)] in latent tokens. Clip-aligned; the last window is pulled back to end on the last frame."""
    win = tokens_for_frames(window_frames)
    if window_frames <= 0 or win >= latent_t:
        return [(0, latent_t)]
    ov = min(tokens_for_frames(overlap_frames), win - CLIP_TOKENS)
    windows = [(0, win)]
    while windows[-1][0] + win < latent_t:
        start = windows[-1][0] + win - ov
        if start + win > latent_t:
            start = (latent_t - win) // CLIP_TOKENS * CLIP_TOKENS
        windows.append((start, min(win, latent_t - start)))
    return windows


def plan_tiles(total, size, overlap):
    """[(start, stop)] in latent units covering [0, total), starts on the 2x2 patch grid."""
    if size <= 0 or size >= total:
        return [(0, total)]
    stride = max(2, size - overlap)
    n = max(2, math.ceil((total - size) / stride) + 1)
    last = (total - size) // 2 * 2
    starts = sorted({min(last, round(i * last / (n - 1) / 2) * 2) for i in range(n)})
    spans = [(s, s + size) for s in starts]
    spans[-1] = (spans[-1][0], total)
    return spans


def feather(n, left, right):
    w = torch.ones(n)
    i = torch.arange(n, dtype=torch.float32)
    if left > 0:
        w = torch.minimum(w, (i + 1.0) / (left + 1.0))
    if right > 0:
        w = torch.minimum(w, (n - i) / (right + 1.0))
    return w


def ease_in_out(n):
    t = torch.linspace(0.0, 1.0, n + 2)[1:-1]
    return t * t * (3.0 - 2.0 * t)


def tile_weight(ys, xs, yi, xi):
    (y0, y1), (x0, x1) = ys[yi], xs[xi]
    wy = feather(y1 - y0, ys[yi - 1][1] - y0 if yi > 0 else 0, y1 - ys[yi + 1][0] if yi + 1 < len(ys) else 0)
    wx = feather(x1 - x0, xs[xi - 1][1] - x0 if xi > 0 else 0, x1 - xs[xi + 1][0] if xi + 1 < len(xs) else 0)
    return wy.view(-1, 1) * wx.view(1, -1)


def crop_box(ys, xs, lat_h, lat_w, rh, rw):
    # the tile's normalized frame box on another grid, snapped to the 2x2 patch grid
    ry0 = min(rh - 2, math.floor(ys[0] / lat_h * rh / 2) * 2)
    ry1 = max(ry0 + 2, min(rh, math.ceil(ys[1] / lat_h * rh / 2) * 2))
    rx0 = min(rw - 2, math.floor(xs[0] / lat_w * rw / 2) * 2)
    rx1 = max(rx0 + 2, min(rw, math.ceil(xs[1] / lat_w * rw / 2) * 2))
    return ry0, ry1, rx0, rx1


def crop_pixels(images, ys, xs, lat_h, lat_w):
    h, w = images.shape[1], images.shape[2]
    y0, y1 = round(ys[0] / lat_h * h), max(round(ys[0] / lat_h * h) + 1, round(ys[1] / lat_h * h))
    x0, x1 = round(xs[0] / lat_w * w), max(round(xs[0] / lat_w * w) + 1, round(xs[1] / lat_w * w))
    return images[:, y0:y1, x0:x1]


# ---------------------------------------------------------------- per-piece conditioning


def ref_window_span(ref_t, k0, k1):
    """Latent range of a reference video covering window [k0, k1); a shorter reference gives its tail."""
    if k1 <= ref_t:
        return k0, k1
    return min(k0, (ref_t - 2) // CLIP_TOKENS * CLIP_TOKENS), ref_t


def window_refs(refs, k0, k1):
    out = []
    for blk in refs:
        if blk["kind"] not in ("video", "video_audio"):
            out.append(blk)
            continue
        s, e = ref_window_span(blk["latent_t"], k0, k1)
        new = dict(blk)
        new["latent"] = blk["latent"][:, :, s:e]
        new["latent_t"] = e - s
        audio = blk.get("audio_latent")
        if audio is not None and blk["ref_audio_t"] > 0:
            a0 = min(audio_ticks(frames_for_tokens(s)), audio.shape[-1])
            a1 = min(a0 + audio_ticks(frames_for_tokens(e) - frames_for_tokens(s)), audio.shape[-1])
            if a1 > a0:
                new["audio_latent"] = audio[..., a0:a1]
                new["ref_audio_t"] = a1 - a0
            else:
                new.update({"audio_latent": None, "ref_audio_t": 0, "kind": "video"})
        out.append(new)
    return out


def tile_refs(refs, ys, xs, lat_h, lat_w, crop_images):
    out = []
    for blk in refs:
        if blk["kind"] not in ("video", "video_audio") and not (crop_images and blk["kind"] == "image"):
            out.append(blk)
            continue
        ry0, ry1, rx0, rx1 = crop_box(ys, xs, lat_h, lat_w, blk["latent_h"], blk["latent_w"])
        new = dict(blk)
        new["latent"] = blk["latent"][..., ry0:ry1, rx0:rx1]
        new["latent_h"], new["latent_w"] = ry1 - ry0, rx1 - rx0
        out.append(new)
    return out


def window_keyframes(keyframes, k0, k1, warned):
    f0, frames = frames_for_tokens(k0), frames_for_tokens(k1) - frames_for_tokens(k0)
    out = []
    for kf in keyframes:
        idx = kf["resolved_frame_index"] - f0
        latent = kf.get("latent")
        span = 1 if latent is None or latent.shape[2] <= 1 else frames_for_tokens(latent.shape[2])
        if idx < 0 or idx + span > frames:
            if idx < frames and idx + span > 0 and "guide_clip" not in warned:
                warned.add("guide_clip")
                logging.warning("MiniMax H3 windows: a guide clip crosses a window boundary and is skipped in that window.")
            continue
        new = dict(kf)
        new["resolved_frame_index"] = idx
        audio = kf.get("audio_latent")
        if audio is not None:
            max_rt = math.floor(audio_ticks(frames) - FRAME_RESCALE * idx)
            new["audio_latent"] = audio[..., :max_rt] if max_rt >= 1 else None
        out.append(new)
    return out


def tile_keyframes(keyframes, ys, xs):
    out = []
    for kf in keyframes:
        if kf.get("latent") is not None:
            kf = dict(kf)
            kf["latent"] = kf["latent"][..., ys[0]:ys[1], xs[0]:xs[1]]
        out.append(kf)
    return out


def piece_text(source, f0, frames, ys, xs, lat_h, lat_w, crop_images, cache):
    """Re-encode the prompt for one piece: <Video N> trimmed to the window (timestamps from 0), cropped to the tile."""
    t0, t1 = f0 / FPS, (f0 + frames) / FPS
    items = []
    for it in source["items"]:
        it = dict(it)
        if it["type"] == "video":
            ts = it.get("timestamps") or [i / 2.0 for i in range(it["data"].shape[0])]
            keep = [i for i, t in enumerate(ts) if t0 <= t < t1] or [min(range(len(ts)), key=lambda i: abs(ts[i] - t0))]
            it["data"] = it["data"][keep]
            it["timestamps"] = [ts[i] - ts[keep[0]] for i in keep]
            if ys is not None:
                it["data"] = crop_pixels(it["data"], ys, xs, lat_h, lat_w)
        elif it["type"] == "image" and ys is not None and crop_images:
            it["data"] = crop_pixels(it["data"], ys, xs, lat_h, lat_w)
        items.append(it)
    clip = source["clip"]
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(source["text"], minimax_ref_items=items))
    return cond[0][0], cond[0][1].get("minimax_token_tags")


def piece_conds(conds, k0, k1, ys, xs, lat_h, lat_w, crop_images, carry, warned, text_cache):
    out = {}
    split_space = ys is not None
    f0, frames = frames_for_tokens(k0), frames_for_tokens(k1) - frames_for_tokens(k0)
    for kind, lst in conds.items():
        new_lst = []
        for c in lst:
            c = c.copy()
            refs = c.get("minimax_refs")
            if refs:
                refs = window_refs(refs, k0, k1)
                c["minimax_refs"] = tile_refs(refs, ys, xs, lat_h, lat_w, crop_images) if split_space else refs
            kfs = window_keyframes(c.get("minimax_keyframes") or [], k0, k1, warned)
            if split_space:
                kfs = tile_keyframes(kfs, ys, xs)
            if carry is not None and kind == "positive":
                kfs = kfs + [carry]
            if kfs or "minimax_keyframes" in c:
                c["minimax_keyframes"] = kfs
            source = c.get(TEXT_SOURCE)
            if source is not None:
                key = (id(source), k0, k1, ys, xs)
                if key not in text_cache:
                    text_cache[key] = piece_text(source, f0, frames, ys, xs, lat_h, lat_w, crop_images, text_cache)
                c["cross_attn"], tags = text_cache[key]
                if tags is not None:
                    c["minimax_token_tags"] = tags
            new_lst.append(c)
        out[kind] = new_lst
    return out


# ---------------------------------------------------------------- the sampling wrapper


class WindowsWrapper:
    def __init__(self, window_frames, overlap_frames, carry_overlap, tile_h, tile_w, tile_overlap, crop_images):
        self.window_frames = window_frames
        self.overlap_frames = overlap_frames
        self.carry_overlap = carry_overlap
        self.tile_h, self.tile_w, self.tile_overlap = tile_h, tile_w, tile_overlap
        self.crop_images = crop_images
        self.warned = set()

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
                 disable_pbar=False, seed=None, latent_shapes=None):
        if latent_shapes is None or len(latent_shapes) < 2 or len(latent_shapes[0]) != 5:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        vshape, ashape = latent_shapes[0], latent_shapes[1]
        latent_t, lat_h, lat_w, audio_t = vshape[2], vshape[3], vshape[4], ashape[-1]
        windows = plan_windows(latent_t, self.window_frames, self.overlap_frames)
        ys = plan_tiles(lat_h, self.tile_h, self.tile_overlap)
        xs = plan_tiles(lat_w, self.tile_w, self.tile_overlap)
        if len(windows) == 1 and len(ys) == 1 and len(xs) == 1:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

        guider = executor.class_obj
        base_conds = {k: [c.copy() for c in v] for k, v in guider.conds.items()}
        tiles = [(yi, xi) for yi in range(len(ys)) for xi in range(len(xs))]
        split_space = len(tiles) > 1

        noise_v, noise_a = comfy.utils.unpack_latents(noise, latent_shapes)[:2]
        lat_v, lat_a = comfy.utils.unpack_latents(latent_image, latent_shapes)[:2]
        masks = comfy.utils.unpack_latents(denoise_mask, latent_shapes)[:2] if denoise_mask is not None else None
        out_v = torch.zeros(vshape, dtype=torch.float32)
        out_a = torch.zeros(ashape, dtype=torch.float32)

        # encode every piece's prompt up front, before the DiT is loaded
        text_cache = {}
        for t0, tn in windows:
            for yi, xi in tiles:
                piece_conds(base_conds, t0, t0 + tn, ys[yi] if split_space else None, xs[xi] if split_space else None,
                            lat_h, lat_w, self.crop_images, None, self.warned, text_cache)
        logging.info("MiniMax H3 windows: {} window(s) x {} tile(s), {} frames{}.".format(
            len(windows), len(tiles), frames_for_tokens(latent_t),
            ", per-piece prompts" if text_cache else " (CLIP not routed through the node: every window reads the full-shot prompt)"))

        done_t = done_a = 0
        steps = sigmas.shape[-1] - 1
        try:
            for wi, (t0, tn) in enumerate(windows):
                a0 = 0 if t0 == 0 else min(audio_ticks(frames_for_tokens(t0)), audio_t)
                a1 = audio_t if wi == len(windows) - 1 else min(audio_t, a0 + audio_ticks(frames_for_tokens(tn)))
                carry_t = 0
                if self.carry_overlap and done_t > t0:
                    carry_t = tokens_for_frames(frames_for_tokens(done_t - t0))
                win_v = torch.zeros(vshape[:2] + (tn,) + tuple(vshape[3:]), dtype=torch.float32)
                win_a = torch.zeros(tuple(ashape[:-1]) + (a1 - a0,), dtype=torch.float32)
                weight = torch.zeros((1, 1, 1, lat_h, lat_w))
                for yi, xi in tiles:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    (y0, y1), (x0, x1) = ys[yi], xs[xi]
                    vs = (slice(None), slice(None), slice(t0, t0 + tn), slice(y0, y1), slice(x0, x1))
                    carry = None
                    if carry_t:
                        carry = {"resolved_frame_index": 0, "latent": out_v[:, :, t0:t0 + carry_t, y0:y1, x0:x1].clone()}
                        a_n = min(done_a - a0, audio_ticks(frames_for_tokens(carry_t)))
                        if a_n > 0:
                            carry["audio_latent"] = out_a[..., a0:a0 + a_n].clone()
                    guider.conds = piece_conds(base_conds, t0, t0 + tn, ys[yi] if split_space else None,
                                               xs[xi] if split_space else None, lat_h, lat_w, self.crop_images,
                                               carry, self.warned, text_cache)
                    shapes = [torch.Size(vshape[:2] + (tn, y1 - y0, x1 - x0)), torch.Size(tuple(ashape[:-1]) + (a1 - a0,))]
                    p_noise = comfy.utils.pack_latents([noise_v[vs], noise_a[..., a0:a1]])[0]
                    p_lat = comfy.utils.pack_latents([lat_v[vs], lat_a[..., a0:a1]])[0]
                    p_mask = None
                    if masks is not None:
                        p_mask = comfy.utils.pack_latents([masks[0].expand(vshape)[vs], masks[1].expand(ashape)[..., a0:a1]])[0]
                    res = executor(p_noise, p_lat, sampler, sigmas, p_mask, None, disable_pbar, seed, latent_shapes=shapes)
                    rv, ra = comfy.utils.unpack_latents(res, shapes)[:2]
                    w = tile_weight(ys, xs, yi, xi).view(1, 1, 1, y1 - y0, x1 - x0)
                    win_v[:, :, :, y0:y1, x0:x1] += rv.float().cpu() * w
                    weight[:, :, :, y0:y1, x0:x1] += w
                    win_a += ra.float().cpu() / len(tiles)  # every tile renders the same soundtrack span
                win_v /= weight

                ov_t = max(0, done_t - t0)
                if ov_t > 0:
                    fade = ease_in_out(ov_t).view(1, 1, -1, 1, 1)
                    out_v[:, :, t0:done_t] = out_v[:, :, t0:done_t] * (1 - fade) + win_v[:, :, :ov_t] * fade
                out_v[:, :, t0 + ov_t:t0 + tn] = win_v[:, :, ov_t:]
                ov_a = max(0, min(done_a, a1) - a0)
                if ov_a > 0:
                    fade = ease_in_out(ov_a)
                    out_a[..., a0:a0 + ov_a] = out_a[..., a0:a0 + ov_a] * (1 - fade) + win_a[..., :ov_a] * fade
                out_a[..., a0 + ov_a:a1] = win_a[..., ov_a:]
                done_t, done_a = max(done_t, t0 + tn), max(done_a, a1)
                logging.info("MiniMax H3 windows: window {}/{} done (frames {}-{}, {} frames carried in).".format(
                    wi + 1, len(windows), frames_for_tokens(t0), frames_for_tokens(t0 + tn) - 1,
                    frames_for_tokens(carry_t) if carry_t else 0))
                if callback is not None:
                    model = guider.model_patcher.model
                    model.latent_shapes = latent_shapes
                    x0 = model.process_latent_in(comfy.utils.pack_latents([out_v, out_a])[0])
                    callback(min(steps - 1, (wi + 1) * steps // len(windows) - 1), x0, x0, steps)
        finally:
            guider.conds = base_conds
        return comfy.utils.pack_latents([out_v, out_a])[0]


def apply(model, wrapper):
    m = model.clone()
    m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY)
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, wrapper)
    return m


def recording_clip(clip):
    """A CLIP clone that records the prompt and reference items it encodes into the conditioning,
    so the windows patch can re-encode the prompt per window / tile."""
    c = clip.clone()
    pending = {}

    def tokenize(text, return_word_ids=False, **kwargs):
        tokens = clip.tokenize(text, return_word_ids=return_word_ids, **kwargs)
        items = kwargs.get("minimax_ref_items")
        if items:
            pending[id(tokens)] = (tokens, {"clip": clip, "text": text, "items": list(items)})
        return tokens

    def encode_from_tokens_scheduled(tokens, *args, **kwargs):
        cond = clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)
        entry = pending.pop(id(tokens), None)
        if entry is not None and entry[0] is tokens:
            cond = [[t[0], dict(t[1], **{TEXT_SOURCE: entry[1]})] for t in cond]
        return cond

    c.tokenize = tokenize
    c.encode_from_tokens_scheduled = encode_from_tokens_scheduled
    return c
