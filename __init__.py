from comfy_api.latest import ComfyExtension, io

from .h3_windows import WindowsWrapper, apply, recording_clip


class MiniMaxH3TemporalWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TemporalWindow",
            display_name="Minimax H3 Temporal Window & Spatial Tiles (mo-akkakk)",
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
                io.Boolean.Input("temporal_windows", default=True,
                                 tooltip="Split the shot into overlapping temporal windows. Off = the whole length runs as one native H3 pass."),
                io.Int.Input("window_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Frames per window (17k+5). Keep it inside what works natively for you, e.g. <= 192."),
                io.Int.Input("overlap_frames", default=22, min=5, max=1000, step=17,
                             tooltip="Frames shared by neighbouring windows (5, 22, 39...). Carried into the next window as its frame-0 guide, then crossfaded."),
                io.Boolean.Input("carry_overlap", default=True,
                                 tooltip="Give each window the previous window's finished overlap frames (video and audio) as its frame-0 guide."),
                io.Boolean.Input("spatial_tiles", default=False,
                                 tooltip="Split each window into overlapping tiles, for stage 1 or stage 2. Each tile follows its crop of the plate. Without a reference video, a full-denoise pass in tiles repeats the subject."),
                io.Int.Input("tile_width", default=960, min=256, max=16384, step=32),
                io.Int.Input("tile_height", default=544, min=256, max=16384, step=32),
                io.Int.Input("tile_overlap", default=256, min=0, max=8192, step=32),
                io.Boolean.Input("crop_ref_images_per_tile", default=True,
                                 tooltip="With tiles, crop <Picture N> to each tile too. On when the picture is a frame of this shot; off for character/style references."),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model, temporal_windows, window_frames, overlap_frames, carry_overlap, spatial_tiles, tile_width, tile_height,
                tile_overlap, crop_ref_images_per_tile, clip=None) -> io.NodeOutput:
        wrapper = WindowsWrapper(window_frames if temporal_windows else 0, overlap_frames, carry_overlap,
                                 tile_height // 16 if spatial_tiles else 0, tile_width // 16 if spatial_tiles else 0,
                                 tile_overlap // 16, crop_ref_images_per_tile)
        return io.NodeOutput(apply(model, wrapper), recording_clip(clip) if clip is not None else None)


class MiniMaxH3WindowExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3TemporalWindow]


async def comfy_entrypoint() -> MiniMaxH3WindowExtension:
    return MiniMaxH3WindowExtension()
