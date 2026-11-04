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
  * optional spatial tiles inside a window, sampled one after another in raster order: each
    tile is prefilled with its finished neighbours over the shared band, whose outer half is
    frozen (denoise 0) and inner half ramps up to full denoise, so seams continue the
    neighbour instead of being averaged; the soundtrack is generated once (first tile) and
    held for the other tiles;
  * optional colour match: a small per-channel DC offset (median, clamped) between a tile
    and its neighbours, and between a window and the previous window over their overlap.

Ideas for the 1.0.4 tile seams, held tile audio, guide resize and colour match come from
LBH-123-AI's MMH3 Split Upscale; cropping guide clips at 5-latent chunks comes from the
native MiniMaxH3ContextWindows support in ComfyUI.
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
COLOR_CLAMP = 0.05


def frames_for_tokens(t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(t))


def tokens_for_frames(frames):
    """Latent length of the largest valid H3 clip (17k + 5 frames) that fits in `frames`."""
    return 2 + 5 * max(0, (frames - 5) // 17)


def audio_ticks(frames):
    return round(frames * FRAME_RESCALE)


def plan_windows(latent_t, window_frames, overlap_frames, min_final_frames=0):
    """[(start, length)] in latent tokens, clip-aligned.

    The last window ends on the last frame. By default it is pulled back to the full window
    length. With min_final_frames > 0 it keeps its natural (shorter) length when that is at
    least min_final_frames, and is otherwise pulled back only far enough to reach that minimum.
    """
    win = tokens_for_frames(window_frames)
    if window_frames <= 0 or win >= latent_t:
        return [(0, latent_t)]
    ov = min(tokens_for_frames(overlap_frames), win - CLIP_TOKENS)
    min_tok = min(win, tokens_for_frames(min_final_frames)) if min_final_frames > 0 else win
    windows = [(0, win)]
    while windows[-1][0] + win < latent_t:
        start = windows[-1][0] + win - ov
        if start + win > latent_t and latent_t - start < min_tok:
            start = (latent_t - min_tok) // CLIP_TOKENS * CLIP_TOKENS
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


def ease_in_out(n):
    t = torch.linspace(0.0, 1.0, n + 2)[1:-1]
    return t * t * (3.0 - 2.0 * t)


def band_profile(n, band):
    """Denoise strength across a tile edge shared with a finished neighbour: outer half frozen (0), inner half ramps to 1."""
    p = torch.ones(n)
    band = min(max(band, 0), n)
    if band <= 0:
        return p, 0, 0
    frozen = band // 2 // 2 * 2
    p[:frozen] = 0.0
    if band > frozen:
        p[frozen:band] = torch.linspace(0.0, 1.0, band - frozen + 2)[1:-1]
    return p, frozen, band


def tile_bands(ys, xs, yi, xi):
    """(top, left) bands shared with tiles already sampled in raster order, in latent units."""
    top = ys[yi - 1][1] - ys[yi][0] if yi > 0 else 0
    left = xs[xi - 1][1] - xs[xi][0] if xi > 0 else 0
    return top, left


def tile_mask(ys, xs, yi, xi):
    (y0, y1), (x0, x1) = ys[yi], xs[xi]
    top, left = tile_bands(ys, xs, yi, xi)
    py, _, _ = band_profile(y1 - y0, top)
    px, _, _ = band_profile(x1 - x0, left)
    return torch.minimum(py.view(-1, 1), px.view(1, -1))


def dc_offset(pairs, clamp=COLOR_CLAMP, min_samples=64):
    """Per-channel median of (new - old) over the given region pairs, clamped; None if there is too little to measure."""
    pairs = [(a, b) for a, b in pairs if a.numel() > 0 and a.numel() // a.shape[1] >= min_samples]
    if not pairs:
        return None
    d = torch.cat([(a.float() - b.float()).movedim(1, -1).reshape(-1, a.shape[1]) for a, b in pairs])
    return d.median(dim=0).values.clamp(-clamp, clamp)


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
    """Frame guides moved to window time. A guide clip that crosses the window edge keeps its 5-latent
    (17-frame) chunks that lie inside the window; its audio is kept as a separate audio guide."""
    f0, frames = frames_for_tokens(k0), frames_for_tokens(k1) - frames_for_tokens(k0)
    a0 = audio_ticks(f0) if k0 > 0 else 0
    a1 = audio_ticks(f0 + frames)
    out = []
    for kf in keyframes:
        idx = kf["resolved_frame_index"] - f0
        latent = kf.get("latent")
        span = 1 if latent is None or latent.shape[2] <= 1 else frames_for_tokens(latent.shape[2])
        if 0 <= idx and idx + span <= frames:
            new = dict(kf)
            new["resolved_frame_index"] = idx
            audio = kf.get("audio_latent")
            if audio is not None:
                max_rt = math.floor(audio_ticks(frames) - FRAME_RESCALE * idx)
                new["audio_latent"] = audio[..., :max_rt] if max_rt >= 1 else None
            out.append(new)
            continue
        if idx >= frames or idx + span <= 0:
            continue
        # crossing the window edge
        if "guide_clip" not in warned:
            warned.add("guide_clip")
            logging.info("MiniMax H3 windows: a guide clip crosses a window edge; the window keeps the part of it inside.")
        if latent is not None and latent.shape[2] > 1:
            vt = latent.shape[2]
            gs = [frames_for_tokens(g) for g in range(vt + 1)]
            chunks = [g for g in range(0, vt, CLIP_TOKENS) if idx + gs[g] >= 0 and idx + gs[min(g + CLIP_TOKENS, vt)] <= frames]
            if chunks:
                new = {k: v for k, v in kf.items() if k != "audio_latent"}
                new["resolved_frame_index"] = idx + gs[chunks[0]]
                new["latent"] = latent[:, :, chunks[0]:chunks[-1] + CLIP_TOKENS]
                out.append(new)
        audio = kf.get("audio_latent")
        if audio is not None:
            t = FRAME_RESCALE * kf["resolved_frame_index"]
            j0, j1 = max(0, math.ceil(a0 - t)), min(audio.shape[-1], math.ceil(a1 - t))
            if j1 > j0:
                new = {k: v for k, v in kf.items() if k != "latent"}
                new["resolved_frame_index"] = (t + j0 - a0) / FRAME_RESCALE
                new["audio_latent"] = audio[..., j0:j1]
                out.append(new)
    return out


def resize_keyframes(keyframes, lat_h, lat_w, warned):
    """Bilinear-resize guide latents encoded at another resolution to the canvas, so tiles crop the right area."""
    out = []
    for kf in keyframes:
        lat = kf.get("latent")
        if lat is not None and tuple(lat.shape[-2:]) != (lat_h, lat_w):
            if "guide_resize" not in warned:
                warned.add("guide_resize")
                logging.info("MiniMax H3 windows: guide latent {}x{} resized to the canvas {}x{}.".format(
                    lat.shape[-1], lat.shape[-2], lat_w, lat_h))
            b, c, t, h, w = lat.shape
            r = torch.nn.functional.interpolate(lat.float().movedim(2, 1).reshape(b * t, c, h, w), size=(lat_h, lat_w),
                                                mode="bilinear", align_corners=False)
            kf = dict(kf)
            kf["latent"] = r.reshape(b, t, c, lat_h, lat_w).movedim(1, 2).to(lat.dtype)
        out.append(kf)
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
                kfs = tile_keyframes(resize_keyframes(kfs, lat_h, lat_w, warned), ys, xs)
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
    def __init__(self, window_frames, overlap_frames, carry_overlap, tile_h, tile_w, tile_overlap, crop_images,
                 min_final_frames=0, tile_audio="generate_once", color_match=False):
        self.window_frames = window_frames
        self.tile_audio = tile_audio
        self.color_match = color_match
        self.overlap_frames = overlap_frames
        self.min_final_frames = min_final_frames
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
        windows = plan_windows(latent_t, self.window_frames, self.overlap_frames, self.min_final_frames)
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
                # finished tiles are written into the canvas; later tiles read their frozen bands from it
                canvas = lat_v[:, :, t0:t0 + tn].float().cpu().clone()
                win_a = None
                for ti, (yi, xi) in enumerate(tiles):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    (y0, y1), (x0, x1) = ys[yi], xs[xi]
                    th, tw = y1 - y0, x1 - x0
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
                    shapes = [torch.Size(vshape[:2] + (tn, th, tw)), torch.Size(tuple(ashape[:-1]) + (a1 - a0,))]

                    v_mask = tile_mask(ys, xs, yi, xi) if split_space else None
                    if v_mask is not None and bool((v_mask >= 1.0).all()):
                        v_mask = None
                    hold_audio = split_space and (self.tile_audio == "keep_input" or ti > 0)
                    a_src = lat_a[..., a0:a1]
                    if hold_audio and self.tile_audio != "keep_input" and win_a is not None:
                        a_src = win_a.to(device=lat_a.device, dtype=lat_a.dtype)

                    p_noise = comfy.utils.pack_latents([noise_v[vs], noise_a[..., a0:a1]])[0]
                    p_lat = comfy.utils.pack_latents([canvas[:, :, :, y0:y1, x0:x1].to(device=lat_v.device, dtype=lat_v.dtype), a_src])[0]
                    p_mask = None
                    if masks is not None or v_mask is not None or hold_audio:
                        dev = masks[0].device if masks is not None else noise_v.device
                        if masks is not None:
                            mv = masks[0].expand(vshape)[vs].float()
                            ma = masks[1].expand(ashape)[..., a0:a1].float().to(dev)
                        else:
                            mv = torch.ones(shapes[0], device=dev)
                            ma = torch.ones(shapes[1], device=dev)
                        if v_mask is not None:
                            mv = mv * v_mask.to(dev).view(1, 1, 1, th, tw)
                        if hold_audio:
                            ma = torch.zeros_like(ma)
                        p_mask = comfy.utils.pack_latents([mv.contiguous(), ma.contiguous()])[0]
                    res = executor(p_noise, p_lat, sampler, sigmas, p_mask, None, disable_pbar, seed, latent_shapes=shapes)
                    rv, ra = comfy.utils.unpack_latents(res, shapes)[:2]
                    rv = rv.float().cpu()

                    if self.color_match and v_mask is not None:
                        old = canvas[:, :, :, y0:y1, x0:x1]
                        top, left = tile_bands(ys, xs, yi, xi)
                        _, fy, by = band_profile(th, top)
                        _, fx, bx = band_profile(tw, left)
                        dc = dc_offset([(rv[:, :, :, fy:by], old[:, :, :, fy:by]), (rv[..., fx:bx], old[..., fx:bx])])
                        if dc is not None:
                            rv = rv - dc.view(1, -1, 1, 1, 1) * v_mask.view(1, 1, 1, th, tw)
                    canvas[:, :, :, y0:y1, x0:x1] = rv
                    if win_a is None:
                        win_a = ra.float().cpu()
                win_v = canvas

                ov_t = max(0, done_t - t0)
                if ov_t > 0 and self.color_match:
                    dc = dc_offset([(win_v[:, :, :ov_t], out_v[:, :, t0:done_t])])
                    if dc is not None:
                        win_v = win_v - dc.view(1, -1, 1, 1, 1)
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
