"""Per-guide release ("until step") for the Multi Guide node.

A guide marked with until_step N is used for the first N sampling steps of each run and left out after that,
so it sets the composition early and leaves the details free. The release happens on every model call:
the H3 payload's guide list is filtered and its guide latents and token layout rebuilt, the same way
ComfyUI's own H3 context windows rebuild them. Every sampling run (each temporal window, each tile, sync
mode) passes through here, so nothing else needs to know about it."""

import torch

import comfy.ldm.minimax.model
import comfy.patcher_extension

UNTIL = "minimax_h3_until"  # on a keyframe: {"latent": steps, "audio_latent": steps}; 0 = never released (every step)
WRAPPER_KEY = "minimax_h3_guide_until"


def tag_new_keyframes(positive, before, video_until, audio_until):
    """Mark the keyframes added since `before` (per-cond keyframe counts) with their until steps."""
    if not video_until and not audio_until:
        return positive
    out = []
    for (t, d), n in zip(positive, before):
        kfs = d.get("minimax_keyframes")
        if kfs is None or len(kfs) <= n:
            out.append([t, d])
            continue
        kfs = list(kfs[:n]) + [dict(kf, **{UNTIL: {"latent": video_until, "audio_latent": audio_until}}) for kf in kfs[n:]]
        out.append([t, dict(d, minimax_keyframes=kfs)])
    return out


def keyframe_counts(positive):
    return [len(d.get("minimax_keyframes") or []) for _, d in positive]


def has_until(positive):
    return any(any(v for v in (kf.get(UNTIL) or {}).values())
               for _, d in positive for kf in (d.get("minimax_keyframes") or []))


def current_step(sigma, transformer_options):
    """0-based step of this model call in its sampling run (sample_sigmas), or None if unknown."""
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None:
        return None
    s = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)
    later = sigmas[1:].float().cpu()
    return int((later >= s * (1 - 1e-5)).sum())


def release(keyframes, step):
    """The keyframes still active at this step (a released part is removed; a keyframe with nothing left is dropped)."""
    out, changed = [], False
    for kf in keyframes:
        until = kf.get(UNTIL)
        if not until:
            out.append(kf)
            continue
        kept = dict(kf)
        for part in ("latent", "audio_latent"):
            n = until.get(part, 0)
            if n and step >= n and kept.get(part) is not None:
                kept[part] = None
                changed = True
        if kept.get("latent") is None and kept.get("audio_latent") is None:
            continue
        out.append(kept)
    return out, changed


class UntilWrapper:
    def __init__(self):
        self.cache = {}

    def __call__(self, executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        payload = kwargs.get("minimax_payload")
        kfs = payload.get("keyframes") if isinstance(payload, dict) else None
        if kfs and any(kf.get(UNTIL) for kf in kfs):
            step = current_step(t, transformer_options)
            if step is not None:
                kept, changed = release(kfs, step)
                if changed:
                    kwargs = dict(kwargs, minimax_payload=self.filtered(payload, kept, step))
        return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def filtered(self, payload, kept, step):
        key = (id(payload), tuple(id(kf) for kf in kept), tuple(
            (kf.get("latent") is not None, kf.get("audio_latent") is not None) for kf in kept))
        hit = self.cache.get(key)
        if hit is not None and hit[0] is payload:
            return hit[1]
        import comfy.model_base
        new = dict(payload)
        new["keyframes"] = kept
        new.update(comfy.model_base.MiniMaxH3._cond_latents(kept, payload.get("refs")))
        layout = payload.get("layout")
        if layout is not None:
            new["layout"] = comfy.ldm.minimax.model.PackedLayout(*layout.signature, keyframes=kept, refs=payload.get("refs"))
        if len(self.cache) > 64:
            self.cache.clear()
        self.cache[key] = (payload, new)
        return new


def apply_until(model):
    m = model.clone()
    m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY)  # one copy is enough
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY, UntilWrapper())
    return m
