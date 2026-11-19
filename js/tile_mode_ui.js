// Minimax H3 V2V and Long Generation nodes (mo-akkakk): show only the settings that belong to the chosen option.
//   tile_mode:  tile_size  -> tile_width / tile_height       tile_count -> tiles_across / tiles_down
//   carry_mode: carry_frames -> overlap_frames               prefix     -> (nothing extra)
// Widgets are only hidden, never removed or reordered, so saved workflows keep their values.
import { app } from "../../scripts/app.js";

const TILE_MODE = {
    tile_size: ["tile_width", "tile_height"],
    tile_count: ["tiles_across", "tiles_down"],
};
const CARRY_MODE = {
    carry_frames: ["overlap_frames"],
    prefix: [],
};
// node class -> { combo widget: { option: widgets shown only with that option } }
const RULES = {
    MiniMaxH3TemporalWindow: { tile_mode: TILE_MODE },
    MiniMaxH3V2V: { tile_mode: TILE_MODE, carry_mode: CARRY_MODE },
    MiniMaxH3LongGeneration: { carry_mode: CARRY_MODE },
};

function setup(node, rules) {
    if (node.__h3OptionUi) return;
    node.__h3OptionUi = true;
    const find = (name) => node.widgets?.find((w) => w.name === name);

    const apply = () => {
        for (const [comboName, byValue] of Object.entries(rules)) {
            const combo = find(comboName);
            if (!combo) continue;
            for (const [value, names] of Object.entries(byValue)) {
                for (const name of names) {
                    const w = find(name);
                    if (w) w.hidden = combo.value !== value;
                }
            }
        }
        const size = node.computeSize();
        node.setSize([Math.max(node.size?.[0] ?? 0, size[0]), size[1]]);
        (node.graph ?? app.graph)?.setDirtyCanvas?.(true, true);
    };

    for (const comboName of Object.keys(rules)) {
        const combo = find(comboName);
        if (!combo) continue;
        const callback = combo.callback;
        combo.callback = function (...args) {
            const r = callback?.apply(this, args);
            apply();
            return r;
        };
    }
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
        const rules = RULES[node.comfyClass];
        if (rules) setup(node, rules);
    },
});
