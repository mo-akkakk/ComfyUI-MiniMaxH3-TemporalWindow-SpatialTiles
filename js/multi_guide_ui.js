// Minimax H3 Multi Guide (mo-akkakk): one frame index and one until_step per guide.
// - frame_idx_N / until_step_N are shown when guide_N is connected (guide_0's always);
//   audio_frame_idx_N / audio_until_step_N when audio_N is connected.
// - These settings belong to their connection: when a guide is disconnected, ComfyUI moves the guides below it
//   up one slot, and their settings move up with them.
// Widgets are only hidden, never removed or reordered, so saved workflows keep their values.
import { app } from "../../scripts/app.js";

const NODE = "MiniMaxH3MultiGuide";
const GROUPS = [
    { slot: /^(?:guides\.)?guide_(\d+)$/, widgets: (i) => [`frame_idx_${i}`, `until_step_${i}`],
      wre: /^(?:frame_idx|until_step)_(\d+)$/, always: 0 },
    { slot: /^(?:audios\.)?audio_(\d+)$/, widgets: (i) => [`audio_frame_idx_${i}`, `audio_until_step_${i}`],
      wre: /^audio_(?:frame_idx|until_step)_(\d+)$/, always: -1 },
];

function setup(node) {
    if (node.__h3MultiGuideUi) return;
    node.__h3MultiGuideUi = true;
    const byLink = new Map(); // link id -> [frame index, until step] of the guide on that link

    const find = (name) => node.widgets?.find((w) => w.name === name);
    const links = (g) => {
        const out = [];
        for (const inp of node.inputs ?? []) {
            const m = g.slot.exec(inp.name);
            if (m && inp.link != null) out.push([Number(m[1]), inp.link]);
        }
        return out;
    };
    const remember = () => {
        const live = new Set();
        for (const g of GROUPS) {
            for (const [i, l] of links(g)) {
                live.add(l);
                byLink.set(l, g.widgets(i).map((name) => find(name)?.value));
            }
        }
        for (const l of [...byLink.keys()]) if (!live.has(l)) byLink.delete(l);
    };

    const apply = () => {
        const used = GROUPS.map(() => new Set());
        GROUPS.forEach((g, k) => {
            for (const [i, l] of links(g)) {
                used[k].add(i);
                // a guide that moved to another slot brings its settings along
                if (!byLink.has(l)) continue;
                g.widgets(i).forEach((name, k) => {
                    const w = find(name);
                    const v = byLink.get(l)[k];
                    if (w && v !== undefined && w.value !== v) w.value = v;
                });
            }
        });
        remember();
        let changed = false;
        for (const w of node.widgets ?? []) {
            GROUPS.forEach((g, k) => {
                const m = g.wre.exec(w.name);
                if (!m) return;
                const i = Number(m[1]);
                const hide = !(i === g.always || used[k].has(i));
                if (w.hidden !== hide) {
                    w.hidden = hide;
                    changed = true;
                }
            });
        }
        if (changed) {
            const size = node.computeSize();
            node.setSize([Math.max(node.size?.[0] ?? 0, size[0]), size[1]]);
        }
        (node.graph ?? app.graph)?.setDirtyCanvas?.(true, true);
    };

    // typing a frame index updates the guide it belongs to
    for (const w of node.widgets ?? []) {
        if (!GROUPS.some((g) => g.wre.test(w.name))) continue;
        const cb = w.callback;
        w.callback = function (...args) {
            const r = cb?.apply(this, args);
            remember();
            return r;
        };
    }
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const r = onConnectionsChange?.apply(this, args);
        setTimeout(apply, 0);
        return r;
    };
    // saved workflows restore links and widget values after the node is created
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const r = onConfigure?.apply(this, args);
        byLink.clear();
        setTimeout(() => { remember(); apply(); }, 0);
        return r;
    };
    apply();
}

app.registerExtension({
    name: "mo-akkakk.MiniMaxH3.MultiGuideUI",
    async nodeCreated(node) {
        if (node.comfyClass === NODE) setup(node);
    },
});
