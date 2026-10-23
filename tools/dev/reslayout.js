// Residual-stream layout (the Llama / Residual MLP convention): the stream runs left → right along the top row through
// the + nodes; each sublayer is a column below its +, flowing up; the main input sits bottom-left; side inputs under the
// column that uses them; the output at the right end of the top row. Runs in the page: (g, sizeOf) → Map id → [x, y].
window.__resLayout = function (g, sizeOf) {
  const ix = NN.index(g), byId = new Map(g.nodes.map(n => [n.id, n]));
  const P = id => (ix.ins.get(id) || []).filter(Boolean).map(k => NN.splitKey(k)[0]);
  const C = id => [...new Set((ix.outs.get(id) || []).map(o => o.to))];
  const memo = new Map(), anc = id => { if (memo.has(id)) return memo.get(id); const s = new Set(); memo.set(id, s); for (const p of P(id)) { s.add(p); anc(p).forEach(a => s.add(a)); } return s; };
  // a nickname caption sits left of its node, so it takes room in the row too
  const capW = n => (n.params.title && NN.isCircle(n) ? n.params.title.length * 6.6 + 14 : 0);
  const constW = n => (n.type === "binop" && n.params.const !== "" && n.params.const != null ? String(n.params.const).replace(/\s+/g, "").length * 6.4 + 22 : 0);
  const W = n => sizeOf(n).w + capW(n), H = n => sizeOf(n).h;
  const adds = g.nodes.filter(n => n.type === 'binop' && n.params.op === '+' && P(n.id).length === 2);
  // main input: the input that reaches the most adds
  const inputs = g.nodes.filter(n => n.type === 'input');
  const main = inputs.slice().sort((a, b) => adds.filter(x => anc(x.id).has(b.id)).length - adds.filter(x => anc(x.id).has(a.id)).length)[0];
  const stream = [main.id], steps = [], used = new Set([main.id]);
  let cur = main.id;
  for (let guard = 0; guard < 20; guard++) {
    // the next residual add: reached from cur, with no other unused add between
    const cand = adds.filter(a => !used.has(a.id) && anc(a.id).has(cur) && P(a.id).every(p => p === cur || anc(p).has(cur) || !anc(p).size || true));
    const A = cand.find(a => !cand.some(b => b !== a && anc(a.id).has(b.id) && anc(b.id).has(cur)));
    if (!A) break;
    const [p0, p1] = P(A.id);
    // the skip: cur itself, or the shorter path from cur (a projection shortcut)
    let skipIn = p0 === cur ? p0 : p1 === cur ? p1 : null, branchIn;
    if (skipIn) branchIn = skipIn === p0 ? p1 : p0;
    else { const n0 = [...anc(p0)].filter(a => anc(a).has(cur) || a === cur).length, n1 = [...anc(p1)].filter(a => anc(a).has(cur) || a === cur).length; skipIn = n0 <= n1 ? p0 : p1; branchIn = skipIn === p0 ? p1 : p0; }
    const desc = id => id === cur || anc(id).has(cur);
    const branch = new Set([...anc(branchIn), branchIn].filter(id => id !== cur && desc(id) && !used.has(id)));
    const shortcut = skipIn === cur ? new Set() : new Set([...anc(skipIn), skipIn].filter(id => id !== cur && desc(id) && !used.has(id) && !branch.has(id)));
    // nodes the branch needs that don't come from the stream (memory, masks, positions, parameters)
    const side = new Set(); for (const id of branch) for (const a of anc(id)) if (!desc(a) && !used.has(a) && a !== main.id) side.add(a);
    [...branch, ...shortcut, ...side, A.id].forEach(id => used.add(id));
    // the stream after the add: single-consumer nodes (post-norm LayerNorm, ReLU) until it forks or ends
    const after = []; let s = A.id;
    for (;;) { const c = C(s); if (c.length !== 1) break; const nx = byId.get(c[0]); if (!nx || nx.type === 'output' || nx.type === 'loss' || (nx.type === 'binop' && nx.params.op === '+')) break; after.push(nx.id); used.add(nx.id); s = nx.id; }
    steps.push({ A: A.id, branch, branchIn, shortcut, side, after });
    cur = s;
  }
  const pos = new Map();
  const TOP = 100, GAPX = 28, GAPY = 34;
  const layerOf = (set, root) => { const d = new Map(); const depth = id => { if (d.has(id)) return d.get(id); d.set(id, 1); let m = 0; for (const p of P(id)) if (set.has(p)) m = Math.max(m, depth(p)); d.set(id, m + 1); return m + 1; }; for (const id of set) depth(id); return d; };
  // a column: rows by depth, deepest just below the top row; returns { width, place(xLeft) }
  const column = set => {
    const d = layerOf(set), maxD = Math.max(0, ...d.values()), rows = [];
    for (const [id, k] of d) (rows[k] = rows[k] || []).push(id);
    const rowW = r => (r || []).reduce((t, id) => t + W(byId.get(id)), 0) + Math.max(0, (r || []).length - 1) * GAPX;
    const width = Math.max(0, ...rows.filter(Boolean).map(rowW));
    // y of each depth, from the top down
    const ys = []; let y = TOP + 13, prevH = 26;
    for (let k = maxD; k >= 1; k--) { const h = Math.max(...rows[k].map(id => H(byId.get(id)))); y += prevH / 2 + GAPY + h / 2; ys[k] = y; prevH = h; }
    const place = xl => {
      const cx = xl + width / 2;
      for (let k = 1; k <= maxD; k++) {
        const r = rows[k].slice(), bary = id => { const ps = P(id).filter(p => pos.has(p)); return ps.length ? ps.reduce((t, p) => t + pos.get(p)[0], 0) / ps.length : byId.get(id).x; };
        r.sort((a, b) => bary(a) - bary(b) || byId.get(a).x - byId.get(b).x);
        let x = cx - rowW(r) / 2; for (const id of r) { const nd = byId.get(id), w = W(nd), c = capW(nd); pos.set(id, [Math.round(x + c + sizeOf(nd).w / 2), Math.round(ys[k])]); x += w + GAPX; }
      }
      return { bottom: ys[1] ?? TOP };
    };
    return { width, place, maxD };
  };
  const mainN = byId.get(main.id);
  const lane = Math.max(W(mainN), ...steps.flatMap(st => [...st.shortcut].map(id => W(byId.get(id)))));
  let X = 60 + W(mainN) / 2 + lane / 2 + 60, lastX = 0, firstBottom = null;
  steps.forEach((st, i) => {
    if (st.shortcut.size) st.shortcutLater = true;
    // what the branch computes from side inputs (cross-attention's keys and values from memory) joins the column;
    // the side inputs themselves (inputs, parameters) go under it
    const sideIn = new Set([...st.side].filter(id => !P(id).length)), colSet = new Set([...st.branch, ...[...st.side].filter(id => !sideIn.has(id))]);
    const c = column(colSet), r = c.place(X); if (firstBottom == null) firstBottom = r.bottom;
    // side inputs in a row under the column, each near its first consumer, without overlapping
    let right = -Infinity;
    const want = id => { const cons = C(id).filter(t => pos.has(t)); return cons.length ? Math.min(...cons.map(t => pos.get(t)[0])) : X; };
    for (const id of [...sideIn].sort((a, b) => want(a) - want(b))) { const w = W(byId.get(id)), cx = Math.max(want(id), right + GAPX + w / 2); pos.set(id, [Math.round(cx), Math.round(r.bottom + 100)]); right = cx + w / 2; }
    const top = pos.get(st.branchIn); let ax = top ? top[0] : X + c.width / 2;
    pos.set(st.A, [ax, TOP]);
    let x = ax + 13; for (const id of st.after) { const w = W(byId.get(id)); x += 40 + w / 2; pos.set(id, [Math.round(x), TOP]); x += w / 2; }
    lastX = x; X = Math.max(X + c.width, x) + 70;
  });
  // the main input: level with the first column's bottom when it feeds one node there, else just below it
  const feeds = C(main.id).filter(t => steps.length && steps[0].branch.has(t)).length;
  pos.set(main.id, [60 + W(mainN) / 2, Math.round((firstBottom ?? TOP + 300) + (feeds > 1 ? 80 : 0))]);
  // a projection shortcut (ResNet's strided 1×1 conv + BatchNorm) rises from the input along the left lane
  for (const st of steps) if (st.shortcutLater) {
    const d = layerOf(st.shortcut), [mx, my] = pos.get(main.id); let y = my, prevH = H(mainN);
    for (let k = 1; k <= Math.max(...d.values()); k++) for (const [id, kk] of d) if (kk === k) { const h = H(byId.get(id)); y -= prevH / 2 + 46 + h / 2; pos.set(id, [Math.round(mx), Math.round(y)]); prevH = h; }
  }
  // outputs (and anything left) to the right on the top row
  let ox = lastX + 50;
  for (const n of g.nodes) if (!pos.has(n.id)) { const w = W(n); pos.set(n.id, [Math.round(ox + w / 2), n.type === 'output' ? TOP : TOP + 80]); ox += w + 40; }
  return [...pos].map(([id, [x, y]]) => ({ id, x, y }));
};
