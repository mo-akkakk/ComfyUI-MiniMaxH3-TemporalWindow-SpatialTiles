// Minimax H3 V2V Temporal Window & Spatial Tiles (mo-akkakk): show only the tile settings of the chosen tile_mode.
// tile_size  -> tile_width / tile_height       tile_count -> tiles_across / tiles_down
// Widgets are only hidden, never removed or reordered, so saved workflows keep their values.
import { app } from "../../scripts/app.js";

const NODE = "MiniMaxH3TemporalWindow";
const BY_MODE = {
    tile_size: ["tile_width", "tile_height"],
    tile_count: ["tiles_across", "tiles_down"],
};

function setup(node) {
    const find = (name) => node.widgets?.find((w) => w.name === name);
    const mode = find("tile_mode");
    if (!mode || mode.__h3TileModeUi) return;
    mode.__h3TileModeUi = true;

    const apply = () => {
        for (const [m, names] of Object.entries(BY_MODE)) {
            for (const name of names) {
                const w = find(name);
                if (w) w.hidden = mode.value !== m;
            }
        }
        const size = node.computeSize();
        node.setSize([Math.max(node.size?.[0] ?? 0, size[0]), size[1]]);
        (node.graph ?? app.graph)?.setDirtyCanvas?.(true, true);
    };

    const callback = mode.callback;
    mode.callback = function (...args) {
        const r = callback?.apply(this, args);
        apply();
        return r;
    };
    // saved workflows set the widget values after the node is created
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const r = onConfigure?.apply(this, args);
        setTimeout(apply, 0);
        return r;
    };
    apply();
}

app.registerExtension({
    name: "mo-akkakk.MiniMaxH3V2V.TileModeUI",
    async nodeCreated(node) {
        if (node.comfyClass === NODE) setup(node);
    },
});
