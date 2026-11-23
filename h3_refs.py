"""Reference to Video with larger reference videos (ref_video_size = match).

The native node scales every reference video to a 768-short-edge canvas (about 1 MP). This node runs the
native node unchanged, but hands it a stand-in VAE: when the native node encodes a reference video, the
stand-in encodes the original video at the generation's pixel area instead (keeping its aspect, never
upscaling the source), so each video is encoded once, directly at the larger size. The text encoder sees
exactly what it sees in the native node. Afterwards the video blocks' sizes are set to match."""

import math

import comfy.utils

CANVAS_MULTIPLE = 32


def match_canvas(vw, vh, width, height):
    """The reference canvas at the generation's pixel area, keeping the video's aspect, never above the source."""
    area = min(width * height, vw * vh)
    s = math.sqrt(area / (vw * vh))
    return (max(CANVAS_MULTIPLE, round(vw * s / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(vh * s / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


class BigVideoVAE:
    """Stand-in for the video VAE during the native Reference to Video call. Images (single frames) and
    anything it does not recognise are encoded as they are; the k-th reference video (5+ frames) is encoded
    from the k-th original video at the larger canvas, when that is larger than the native one."""

    def __init__(self, vae, videos, width, height):
        self.vae = vae
        self.videos = list(videos)  # originals, in the order the native node encodes them
        self.width, self.height = width, height
        self.sizes = []             # per encoded video: (w, h) actually used

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, pixels, *args, **kwargs):
        if pixels.ndim != 4 or pixels.shape[0] < 5 or len(self.sizes) >= len(self.videos):
            return self.vae.encode(pixels, *args, **kwargs)
        orig = self.videos[len(self.sizes)]
        n, nh, nw = pixels.shape[0], pixels.shape[1], pixels.shape[2]
        cw, ch = match_canvas(orig.shape[2], orig.shape[1], self.width, self.height)
        if cw * ch <= nw * nh:
            self.sizes.append((nw, nh))  # the native canvas is already as large: keep it
            return self.vae.encode(pixels, *args, **kwargs)
        big = comfy.utils.common_upscale(orig[:n, ..., :3].movedim(-1, 1), cw, ch, "lanczos", "disabled").movedim(1, -1)
        self.sizes.append((cw, ch))
        return self.vae.encode(big, *args, **kwargs)


def set_video_block_sizes(cond, sizes):
    """Give the video blocks (in order) the canvas sizes their latents were encoded at."""
    out = []
    for t, d in cond:
        blocks = list(d.get("minimax_refs") or [])
        vid = [i for i, b in enumerate(blocks) if b.get("kind") in ("video", "video_audio")]
        for i, (w, h) in zip(vid, sizes):
            blocks[i] = dict(blocks[i], latent_w=w // 16, latent_h=h // 16)
        out.append([t, dict(d, minimax_refs=blocks)] if blocks else [t, d])
    return out
