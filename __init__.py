from comfy_api.latest import ComfyExtension, io

from .h3_windows import WindowsWrapper, apply, parse_schedule, recording_clip


class MiniMaxH3TemporalWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TemporalWindow",
            display_name="Minimax H3 V2V Temporal Window & Spatial Tiles (mo-akkakk)",
            category="model/patch/minimax",
            search_aliases=["h3 temporal window", "h3 spatial tiles", "h3 long video", "minimax overlap", "context windows", "mo-akkakk"],
            description="Put after the model, and route the CLIP through it into Reference to Video. "
                        "Samples the video as overlapping temporal windows, one after another: each window is a normal H3 run on its "
                        "slice of the plate, starts from the previous window's finished overlap frames (frame-0 guide), and the "
                        "overlap is crossfaded. Optional spatial tiles, each following its crop of the plate. Works with KSampler and SamplerCustomAdvanced.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True,
                              tooltip="Route the CLIP through here into Reference to Video so each window re-encodes the prompt with only "
                                      "its own slice of <Video N> (timestamps from 0), like a manual batch. Strongly recommended."),
                io.Image.Input("plate", optional=True,
                               tooltip="Optional, for tiles: the full-resolution version of the video you feed as ref_video_0 "
                                       "(same frames, same order, at least as long). Each tile's plate piece is then cut from it "
                                       "exactly and at full detail, instead of from Reference to Video's 768p copy. Needs vae. "
                                       "Feed it at about your output resolution (a 4K plate works but uses a lot of RAM). See plate_scale."),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE, used with plate to encode each tile's plate piece."),
                io.Boolean.Input("temporal_windows", default=True,
                                 tooltip="Split the shot into overlapping temporal windows. Off = the whole length runs as one native H3 pass."),
                io.Int.Input("window_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Frames per window (17k+5). Keep it inside what works natively for you, e.g. <= 192."),
                io.Int.Input("overlap_frames", default=22, min=5, max=1000, step=17,
                             tooltip="Frames shared by neighbouring windows (5, 22, 39...). The next window regenerates them and they are crossfaded; carry_frames of them guide the next window."),
                io.Boolean.Input("short_final_window", default=False,
                                 tooltip="Let the last window be shorter than window_frames instead of pulling it back to full length. "
                                         "Saves the frames that would be generated twice at the end, which matters most with large windows. Off = previous behaviour."),
                io.Int.Input("min_final_window_frames", default=22, min=5, max=3600, step=17,
                             tooltip="With short_final_window on: the last window is never shorter than this (17k+5: 5, 22, 39, 56...). "
                                     "If the natural tail is shorter, the window is pulled back only far enough to reach it."),
                io.Boolean.Input("carry_overlap", default=True,
                                 tooltip="Give each window the previous window's finished overlap frames (video and audio) as a guide (see carry_frames)."),
                io.Int.Input("carry_frames", default=22, min=5, max=1000, step=17,
                             tooltip="With carry_overlap: how many frames of the overlap the next window gets as its guide (5, 22, 39...), "
                                     "taken right before its new frames. Equal to or above overlap_frames = the whole overlap (as before 1.2.0). "
                                     "Fewer = less extra memory from window 2 on, while the whole overlap is still crossfaded. "
                                     "Capped at overlap_frames."),
                io.Boolean.Input("style_anchor", default=False,
                                 tooltip="Against the look fading over a long shot: every window after the first also gets the first "
                                         "finished frame as one more <Picture> reference, with a line asking every frame to keep its "
                                         "lighting, weather, colours and styling. Motion still follows <Video 1>. Needs vae and the CLIP "
                                         "routed through this node into Reference to Video. The later prompts are encoded after window 1."),
                io.Boolean.Input("spatial_tiles", default=False,
                                 tooltip="Split each window into overlapping tiles, for stage 1 or stage 2. Each tile follows its crop of the plate. Without a reference video, a full-denoise pass in tiles repeats the subject."),
                io.Combo.Input("tile_mode", options=["tile_size", "tile_count"], default="tile_size",
                               tooltip="How tiles are set. tile_size: use tile_width / tile_height (pixels). tile_count: split the frame "
                                       "into tiles_across x tiles_down tiles; their size follows from the frame size. Both use tile_overlap."),
                io.Int.Input("tile_width", default=960, min=256, max=16384, step=32,
                             tooltip="Tile width in pixels (tile_mode = tile_size)."),
                io.Int.Input("tile_height", default=544, min=256, max=16384, step=32,
                             tooltip="Tile height in pixels (tile_mode = tile_size)."),
                io.Int.Input("tiles_across", default=2, min=1, max=16,
                             tooltip="With tile_mode = tile_count: number of tiles horizontally (1 = no horizontal split)."),
                io.Int.Input("tiles_down", default=2, min=1, max=16,
                             tooltip="With tile_mode = tile_count: number of tiles vertically (1 = no vertical split)."),
                io.Int.Input("tile_overlap", default=256, min=0, max=8192, step=32,
                             tooltip="Pixels shared by neighbouring tiles, used in both tile modes."),
                io.Boolean.Input("tile_sync", default=False,
                                 tooltip="Tiles take every sampling step together and their overlaps are averaged after each step, "
                                         "so neighbouring tiles agree on where things are: no ghosting where a moving subject crosses "
                                         "a seam. Same VRAM (the model still sees one tile at a time) and about the same time. "
                                         "Off = each tile is sampled on its own and blended at the end (1.2.0 and earlier)."),
                io.Combo.Input("tile_carry", options=["own_tile", "blended"], default="own_tile",
                               tooltip="With tiles and carry_overlap: own_tile = each tile continues from its own unblended frames of the "
                                       "previous window, so blending errors are not fed forward (less drift over long shots). "
                                       "blended = continue from the blended picture (1.0.6 and earlier)."),
                io.Combo.Input("tile_audio", options=["generate_once", "keep_input"], default="generate_once",
                               tooltip="With tiles: generate_once = use the soundtrack generated by the first tile (instead of averaging one per tile). "
                                       "keep_input = output the incoming audio latent untouched (stage 2, when stage 1's audio is final)."),
                io.Boolean.Input("crop_ref_images_per_tile", default=True,
                                 tooltip="With tiles, crop <Picture N> to each tile too. On when the picture is a frame of this shot; off for character/style references."),
                io.Boolean.Input("color_match", default=False,
                                 tooltip="Remove a small colour/brightness offset (per-channel median, clamped) between a tile and the "
                                         "tiles already blended into its overlap, and between a window and the previous one over their overlap. Off = no correction."),
                io.Float.Input("plate_scale", default=1.0, min=0.25, max=1.0, step=0.05,
                               tooltip="Only with the plate input connected. Size of each tile's plate piece relative to the tile. "
                                       "1.0 = full detail (slowest: about 4x the reference tokens of no plate). 0.5 = a quarter of the "
                                       "reference tokens, much faster, still exactly aligned and sharper than without a plate."),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model, temporal_windows, window_frames, overlap_frames, short_final_window, min_final_window_frames,
                carry_overlap, carry_frames, style_anchor, spatial_tiles, tile_mode, tile_width, tile_height, tiles_across,
                tiles_down, tile_overlap, tile_sync, tile_carry, tile_audio, crop_ref_images_per_tile, color_match, plate_scale,
                clip=None, plate=None, vae=None) -> io.NodeOutput:
        wrapper = WindowsWrapper(window_frames if temporal_windows else 0, overlap_frames, carry_overlap,
                                 tile_height // 16 if spatial_tiles else 0, tile_width // 16 if spatial_tiles else 0,
                                 tile_overlap // 16, crop_ref_images_per_tile,
                                 min_final_window_frames if short_final_window else 0,
                                 tile_audio, color_match, tile_carry, plate if spatial_tiles else None, vae, plate_scale,
                                 tile_counts=(tiles_across, tiles_down) if spatial_tiles and tile_mode == "tile_count" else None,
                                 carry_frames=carry_frames if carry_frames < overlap_frames else 0,
                                 tile_sync=tile_sync, style_anchor=style_anchor)
        return io.NodeOutput(apply(model, wrapper), recording_clip(clip) if clip is not None else None)


class MiniMaxH3LongGeneration(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongGeneration",
            display_name="Minimax H3 Long Generation (mo-akkakk)",
            category="model/patch/minimax",
            search_aliases=["h3 long generation", "h3 prompt schedule", "h3 long t2v", "h3 long i2v", "minimax long video", "mo-akkakk"],
            description="For text-to-video, image-to-video and first/last-frame generation (no reference video). Put after the model, "
                        "and route the CLIP through it into Image to Video or Reference to Video. The video is generated as overlapping "
                        "windows, one after another, each continuing from the previous one, with an optional prompt schedule so the "
                        "story moves on instead of restarting every window. For video-to-video use the V2V node instead.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True,
                              tooltip="Route the CLIP through here into Image to Video or Reference to Video. Needed for the prompt schedule "
                                      "and for keeping the first/last image only in its own window."),
                io.String.Input("prompt_schedule", multiline=True, default="",
                                tooltip="Optional timeline for the whole video, one line per action, in seconds (H3 runs at 24 fps):\n"
                                        "0s: he walks along the street\n6s: he stops at the bakery window\n"
                                        "or H3's own format: [0s-6s] he walks along the street\n"
                                        "Write the look, character and setting once in the conditioning node's prompt; it is kept for "
                                        "every window. Each window then gets that scene plus the part of this timeline it covers, "
                                        "re-timed so the window starts at 0 s. Empty = a 'Timeline:' written in the conditioning "
                                        "prompt is re-timed the same way; with none, every window uses the prompt as it is."),
                io.Boolean.Input("continuous_shot", default=True,
                                 tooltip="Add 'One continuous shot, no cuts.' to every window's prompt, so H3 continues the carried "
                                         "frames instead of cutting to a new shot when the action changes. Turn off if you want cuts."),
                io.Int.Input("window_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Frames per window (17k+5). 124 = 5.2 s. Keep it inside what works natively for you."),
                io.Int.Input("overlap_frames", default=39, min=5, max=1000, step=17,
                             tooltip="Frames shared by neighbouring windows (5, 22, 39...). Without a reference video the overlap is "
                                     "what carries the motion forward, so use more than for V2V (39 recommended)."),
                io.Boolean.Input("short_final_window", default=False,
                                 tooltip="Let the last window be shorter than window_frames instead of pulling it back to full length."),
                io.Int.Input("min_final_window_frames", default=22, min=5, max=3600, step=17,
                             tooltip="With short_final_window on: the last window is never shorter than this (17k+5: 5, 22, 39, 56...)."),
                io.Boolean.Input("carry_overlap", default=True,
                                 tooltip="Each window starts from the previous window's finished overlap frames (video and audio). "
                                         "Keep it on: without it, windows are unrelated clips."),
                io.Combo.Input("first_last_images", options=["own_window", "all_windows"], default="own_window",
                               tooltip="Image to Video's first/last frame images in the prompt. own_window: the first image is only in the "
                                       "first window's prompt and the last image only in the last window's (matches where they are "
                                       "used as frames). all_windows: every window's prompt shows them (can help keep a character's "
                                       "look, but later windows may drift back toward the first frame)."),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model, prompt_schedule, continuous_shot, window_frames, overlap_frames, short_final_window,
                min_final_window_frames, carry_overlap, first_last_images="own_window", clip=None) -> io.NodeOutput:
        wrapper = WindowsWrapper(window_frames, overlap_frames, carry_overlap, 0, 0, 0, False,
                                 min_final_window_frames if short_final_window else 0,
                                 schedule=parse_schedule(prompt_schedule), image_policy=first_last_images,
                                 continuous=continuous_shot)
        return io.NodeOutput(apply(model, wrapper), recording_clip(clip, record_all=True) if clip is not None else None)


class MiniMaxH3WindowExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3TemporalWindow, MiniMaxH3LongGeneration]


WEB_DIRECTORY = "./js"  # tile_mode_ui.js: shows only the tile settings of the chosen tile_mode


async def comfy_entrypoint() -> MiniMaxH3WindowExtension:
    return MiniMaxH3WindowExtension()
