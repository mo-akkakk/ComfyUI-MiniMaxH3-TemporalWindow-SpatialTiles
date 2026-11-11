// Minimax H3 Multi Guide (mo-akkakk): one frame index per guide.
// - frame_idx_N is shown when guide_N is connected (frame_idx_0 always); audio_frame_idx_N when audio_N is connected.
// - Each frame index belongs to its connection: when a guide is disconnected, ComfyUI moves the guides below it
//   up one slot, and their frame indexes move up with them.
// Widgets are only hidden, never removed or reordered, so saved workflows keep their values.
import { app } from "../../scripts/app.js";

const NODE = "MiniMaxH3MultiGuide";
const GROUPS = [
    { slot: /^(?:guides\.)?guide_(\d+)$/, widget: (i) => `frame_idx_${i}`, wre: /^frame_idx_(\d+)$/, always: 0 },
    { slot: /^(?:audios\.)?audio_(\d+)$/, widget: (i) => `audio_frame_idx_${i}`, wre: /^audio_frame_idx_(\d+)$/, always: -1 },
];

function setup(node) {
    if (node.__h3MultiGuideUi) return;
    node.__h3MultiGuideUi = true;
    const byLink = new Map(); // link id -> frame index of the guide on that link

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
                const w = find(g.widget(i));
                if (w) byLink.set(l, w.value);
            }
        }
        for (const l of [...byLink.keys()]) if (!live.has(l)) byLink.delete(l);
    };

    const apply = () => {
        const used = GROUPS.map(() => new Set());
        GROUPS.forEach((g, k) => {
            for (const [i, l] of links(g)) {
                used[k].add(i);
                // a guide that moved to another slot brings its frame index along
                const w = find(g.widget(i));
                if (w && byLink.has(l) && w.value !== byLink.get(l)) w.value = byLink.get(l);
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
