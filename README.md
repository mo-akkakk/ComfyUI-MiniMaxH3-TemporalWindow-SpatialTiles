![Minimax H3 Temporal Window & Spatial Tiles node](node.png)

# Minimax H3 Temporal Window & Spatial Tiles

by Mohamed ([mo-akkakk](https://registry.comfy.org/publishers/mo-akkakk))

## What's new

**1.0.4**
- Tiles now run one after another. Each tile is prefilled with its finished neighbours over the overlap: the outer half of that band is frozen and the inner half is blended in, so seams continue the neighbour instead of being averaged.
- New `tile_audio` setting. `generate_once` (default): the first tile makes the soundtrack and the other tiles keep it, instead of averaging one soundtrack per tile. `keep_input`: the incoming audio is kept untouched (stage 2).
- A guide clip that crosses a window edge is no longer dropped: the window keeps the 17-frame chunks inside it, plus the clip's audio.
- Guide latents at another resolution are resized to the frame before tiles crop them.
- New `color_match` switch (off by default): removes a small colour/brightness offset between tiles and between windows.
- Saved workflows keep their settings; the two new widgets are added at the end.

**1.0.3**
- New `short_final_window` switch (off by default): the last window can keep its natural shorter length instead of being pulled back to a full window.
- New `min_final_window_frames` (default 22): the shortest the last window may be when `short_final_window` is on.

**1.0.2**
- README and tooltips: spatial tiles also work in stage 1 when a plate (reference video) drives the shot.
- Removed the note that the plate must be 24 fps; testing showed it doesn't matter.

**1.0.1**
- New `temporal_windows` switch: turn windows off to run the whole length as one native pass (tiles still available).

**1.0.0**
- First release: sequential temporal windows with overlap carry and crossfade, per-window (and per-tile) prompt re-encoding, spatial tiles, works with KSampler and SamplerCustomAdvanced.

Long video-to-video for the **open-weights MiniMax H3** in native ComfyUI. It is one node that goes after the model.

- **Temporal windows (stage 1 and stage 2).** The shot is sampled as overlapping windows, **one after another**. Each window is an ordinary native H3 run on its own slice of the plate. It starts from the previous window's finished overlap frames, video and audio, which are given to it as a frame-0 guide. The overlap is then crossfaded. This keeps every window inside the length where H3 tracks the plate accurately, while the result runs as long as you like.
- **Spatial tiles (stage 1 and stage 2).** Each window can be split into overlapping tiles. Each tile runs on its own crop of the plate, with the prompt re-encoded for that crop. Tiles run one after another: each tile is prefilled with its finished neighbours over the overlap, the outer half of that band is frozen and the inner half is blended in, so the seam continues the neighbour instead of being averaged. The soundtrack is generated once and held across tiles. This works for a full-denoise first pass as well as a stage-2 refine, because every tile is anchored to its region of the plate.

Nothing inside the model is patched, and every window and tile goes through the native H3 conditioning and sampling code. With a single window and no tiles, the output is bit-identical to the plain native workflow.

## Install

**ComfyUI Manager / Registry:** search for *Minimax H3 Temporal Window & Spatial Tiles*.

**Manual:**
```
cd ComfyUI/custom_nodes
git clone https://github.com/mo-akkakk/ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles
```
Restart ComfyUI. It needs ComfyUI with native MiniMax H3 support (v0.35.0 or newer) and has no extra dependencies.

## Wiring

```
UNETLoader -> (shift / LoRA / low-VRAM patches) -> Minimax H3 Temporal Window & Spatial Tiles -> BasicGuider or KSampler
CLIPLoader ------------------------------------> (clip in ... clip out) -> MiniMax H3 Reference to Video
```

Everything else is the stock H3 workflow: Reference to Video, Add Guide, SamplerCustomAdvanced or KSampler, and VAE Decode.

- **Stage 1:** full sigmas. Turn `spatial_tiles` on when the frame is bigger than what H3 handles well on your setup.
- **Stage 2:** decode → upscale → VAE Encode → Concat AV Latent (with the stage-1 audio), or upscale the latent directly with [LBH-123-AI's MiniMax H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler), which skips the decode/encode round trip. Then use a second copy of the node, usually with `spatial_tiles` on, and sample with the low part of SplitSigmas.

**Route the CLIP through the node.** It records what Reference to Video encodes and re-encodes the prompt per window, and per tile. Qwen then sees only that window's frames of `<Video N>`, with timestamps from 0, exactly like a manually cut batch. Without this, every window reads a description of the whole shot, and timing drifts.

## Settings

| Input | Meaning |
|---|---|
| `temporal_windows` | Split the shot into overlapping windows. Off = the whole length runs as one pass. |
| `window_frames` | Frames per window, on H3's 17k+5 grid (124 = 5.2 s, 175 = 7.3 s). Keep it within what H3 handles accurately on your setup. |
| `overlap_frames` | Frames shared by neighbouring windows (5, 22, 39…). They are carried into the next window as its frame-0 guide, then crossfaded. The last window is pulled back to end on the last frame, so its overlap can be larger. |
| `short_final_window` | Off (default): the last window is pulled back so it is a full `window_frames` long, which re-generates some frames at the end. On: the last window keeps its natural, shorter length, which saves those frames — most useful with large windows. |
| `min_final_window_frames` | With `short_final_window` on: the shortest the last window may be (17k+5: 5, 22, 39, 56…). A shorter tail is pulled back only far enough to reach it. |
| `carry_overlap` | Carry the finished overlap (video and audio) into the next window. |
| `spatial_tiles` | Split each window into overlapping tiles (stage 1 or stage 2). Needs a plate: without a reference video, a full-denoise pass in tiles repeats the subject in every tile. |
| `tile_width` / `tile_height` / `tile_overlap` | Tile size in pixels. Use a size H3 handles well on your setup, with ≥ 192 px overlap. |
| `crop_ref_images_per_tile` | Crop `<Picture N>` to each tile as well. Turn it on when the picture is a frame of the shot (e.g. an edited first frame). Turn it off for character, product or style references. |
| `tile_audio` | With tiles. `generate_once` (default): the first tile generates the soundtrack and every other tile keeps it unchanged. `keep_input`: the incoming audio latent is kept untouched in every tile — use it in stage 2 when the stage-1 audio is final. |
| `color_match` | Off by default. Removes a small colour/brightness offset (per-channel median, clamped to ±0.05 in latent space) between a tile and its finished neighbours, and between a window and the previous window over their overlap. Try it if you see a slight tone step at seams. |

Reference videos (the plate) are always sliced per window and cropped per tile, because they are frame-aligned with the output. Frame guides from Add Guide go only into the window that contains their frame. A guide clip that crosses a window edge keeps the 17-frame chunks that lie inside the window, and its audio is kept as a separate audio guide. Guide latents encoded at another resolution are resized to the canvas before tiles crop them.

## Tips

- Write the prompt as true for **every** frame, because each window reads it on its own.
- If you see a seam where windows join, increase `overlap_frames` (22 → 39).
- If you see a seam between tiles, increase `tile_overlap` (the frozen half and the blended half both grow with it), then try `color_match`.

## How it works

The node registers an `OUTER_SAMPLE` wrapper on the model, so it works with any guider and sampler. For each window it does the following:

1. It slices the noise and latent to the window, and the audio to the matching span on H3's 40 Hz audio grid.
2. It re-times the native conditioning to the window. Reference-video latents are sliced, and keyframes are moved to window time.
3. It appends the previous window's finished overlap as a keyframe clip at frame 0.
4. It swaps in the per-window prompt encoding.
5. It runs the normal sampling call.

Windows start on H3's 17-frame VAE-clip boundaries, so every window and every carried overlap is a valid H3 length.

## Tests

```
python custom_nodes/ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles/tests/test_windows.py
```

Run this from the ComfyUI root. The tests run ComfyUI's real H3 conditioning, model and sampler code on CPU, with a tiny random-weight DiT and recording stand-ins for Qwen and the VAE. They check that:

- a single window is bit-identical to native sampling;
- each window gets its own slice of the plate;
- the overlap carry is correct;
- the per-window prompt shows only that window's frames, with timestamps from 0;
- it works through KSampler;
- tiles work with full and partial sigmas;
- each tile's band with finished neighbours is prefilled and frozen, and the soundtrack is generated once (or kept from the input);
- guide clips crossing a window edge are cropped at 17-frame chunks, and guides at another resolution are resized.

## Limits

- Tiles still each see only their own crop, so very large flat areas (sky, walls) can drift slightly in tone across a big frame. `color_match` and a wider `tile_overlap` help.
- Batch size 1, the same as native H3.

## Credits

The 1.0.4 tile seams (sequential tiles with a frozen, prefilled overlap), the held tile audio, the guide resize and the colour match are adapted from ideas in [LBH-123-AI's MMH3 Split Upscale](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) (MIT). Cropping guide clips at 17-frame chunks follows the native MiniMaxH3ContextWindows support in ComfyUI.

## License

MIT, © 2026 Mohamed ([mo-akkakk](https://registry.comfy.org/publishers/mo-akkakk)).

