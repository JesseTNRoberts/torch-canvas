// ───────────────────────── NN Builder core ─────────────────────────
// Graph model, symbolic shape inference, PyTorch code generation,
// Python (PyTorch subset) importer, torch.fx JSON importer, auto-layout.
// Runs in the browser and in Node (module.exports at the bottom).
const NN = (() => {
  'use strict';
  let _seq = 0;
  const uid = (p = 'n') => p + (++_seq).toString(36) + Math.random().toString(36).slice(2, 6);

  const BATCH = 'e';
  const PY_KW = new Set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self torch nn F'.split(' '));

  // ── node type registry ──
  const DEFAULTS = {
    input: { name: 'x', shape: 'e, d_in', title: '' },
    output: { name: 'out', title: '' },
    linear: { in: 'f', out: 'f', bias: false, init: 'default', init_arg: '', zero_bias: true, tie: '', name: '', title: '', var: '' },
    norm: { kind: 'RMSNorm', dim: 'f', groups: 8, eps: '', name: '', title: '', var: '' },
    act: { fn: 'silu', dim: '-1', title: '', var: '' },
    dropout: { p: 0.1, name: '', title: '', var: '' },
    binop: { op: '+', const: '', rev: false, title: '', var: '' },
    func: { fn: 'torch.cat', extra: 'dim=-1', n: 2, init: '', name: '', title: '', var: '' },
    block: { block: '', args: {}, nIn: 1, repeat: 1, name: '', title: '', var: '' },
    // token or position embedding: a learned table of `num` vectors of size `dim`
    embed: { num: 'vocab', dim: 'd', positions: false, init: 'default', init_arg: '', tie: '', name: '', title: '', var: '' },
    // attention plumbing (no weights)
    heads: { heads: 'n_head', title: '', var: '' },   // (…, T, C) → (…, H, T, C/H)
    merge: { title: '', var: '' },                    // (…, H, T, D) → (…, T, H·D)
    transpose: { title: '', var: '' },                // swap the last two dims (Kᵀ)
    scale: { title: '', var: '' },                    // × 1/√(last dim)
    mask: { mode: 'causal', title: '', var: '' },     // causal mask: future positions → −inf; 'given': an allowed mask comes in on input 2
    rope: { base: '10000', mode: 'seq', sections: '16, 24, 24', title: '', var: '' }, // rotary position embedding on (…, T, D); 'axial'/'mrope': positions on input 2
    // loss: port 0 = a model output (raw, before any final activation), port 1 = the target
    loss: { fn: 'cross_entropy', mode: 'given', label_smoothing: '', ignore_index: '', delta: '1.0', weight: '1', mask_prob: '0.15', mask_id: '', title: '' },
  };
  const ACTS = ['sigmoid', 'silu', 'relu', 'gelu', 'gelu_tanh', 'tanh', 'softmax'];
  // Loss functions. act = what predict() applies at inference (the loss works on the raw output).
  const LOSSES = {
    cross_entropy: { label: 'Cross-entropy', torch: 'F.cross_entropy', act: 'softmax', note: 'Classes or tokens; takes logits' },
    bce: { label: 'Binary cross-entropy', torch: 'F.binary_cross_entropy_with_logits', act: 'sigmoid', note: 'Yes/no per output; takes logits' },
    mse: { label: 'Mean squared error', torch: 'F.mse_loss', act: null, note: 'Regression' },
    l1: { label: 'L1 (mean absolute error)', torch: 'F.l1_loss', act: null, note: 'Regression, robust to outliers' },
    huber: { label: 'Huber', torch: 'F.huber_loss', act: null, note: 'MSE near zero, L1 far away' },
  };
  // Normalization layers. channelFirst: PyTorch expects (N, C, L), so (e, t, d) inputs are transposed around the call.
  const NORMS = {
    LayerNorm: { label: 'LayerNorm', note: 'Per example, over the features', short: 'LayerNorm' },
    RMSNorm: { label: 'RMSNorm', note: 'LayerNorm without mean-centering or bias', short: 'RMSNorm' },
    BatchNorm1d: { label: 'BatchNorm', note: 'Per feature, over the batch (running stats)', short: 'BatchNorm', channelFirst: true },
    GroupNorm: { label: 'GroupNorm', note: 'Per example, over groups of features', short: 'GroupNorm', channelFirst: true },
    InstanceNorm1d: { label: 'InstanceNorm', note: 'Per example and feature, over the sequence', short: 'InstanceNorm', channelFirst: true },
  };
  const normCtor = (p, d) => {
    const eps = p.eps !== '' && p.eps != null ? `, eps=${p.eps}` : '';
    switch (p.kind) {
      case 'LayerNorm': return `nn.LayerNorm(${d}${eps})`;
      case 'BatchNorm1d': return `nn.BatchNorm1d(${d}${eps})`;
      case 'GroupNorm': return `nn.GroupNorm(${p.groups || 8}, ${d}${eps})`;
      case 'InstanceNorm1d': return `nn.InstanceNorm1d(${d}${eps})`;
      default: return `nn.RMSNorm(${d}${eps})`;
    }
  };
  // Weight initializers for nn.Linear. arg = the knob shown in the UI (std or gain).
  const INITS = {
    default: { label: 'PyTorch default', note: 'Kaiming uniform, a=√5', arg: null },
    xavier_uniform: { label: 'Xavier uniform', note: 'Glorot; good for tanh/sigmoid', arg: 'gain', def: '1' },
    xavier_normal: { label: 'Xavier normal', note: 'Glorot, Gaussian', arg: 'gain', def: '1' },
    kaiming_uniform: { label: 'Kaiming uniform', note: 'He; for ReLU-family', arg: null },
    kaiming_normal: { label: 'Kaiming normal', note: 'He, Gaussian', arg: null },
    orthogonal: { label: 'Orthogonal', note: 'Preserves norms', arg: 'gain', def: '1' },
    normal: { label: 'Normal', note: 'N(0, std²)', arg: 'std', def: '0.02' },
    trunc_normal: { label: 'Truncated normal', note: 'N(0, std²) clipped at ±2·std', arg: 'std', def: '0.02' },
    zeros: { label: 'Zeros', note: 'Block starts as identity when on a residual output', arg: null },
  };
  function initCode(p, W) {
    const k = p.init || 'default', spec = INITS[k]; if (!spec || k === 'default') return null;
    const a = p.init_arg !== '' && p.init_arg != null ? String(p.init_arg) : spec.def;
    switch (k) {
      case 'xavier_uniform': case 'xavier_normal': case 'orthogonal': return `nn.init.${k}_(${W}${a && +a !== 1 ? `, gain=${a}` : ''})`;
      case 'kaiming_uniform': case 'kaiming_normal': return `nn.init.${k}_(${W}, nonlinearity="relu")`;
      case 'normal': return `nn.init.normal_(${W}, std=${a})`;
      case 'trunc_normal': { const t = +(2 * +a).toPrecision(6); return `nn.init.trunc_normal_(${W}, std=${a}, a=-${t}, b=${t})`; }
      case 'zeros': return `nn.init.zeros_(${W})`;
    }
    return null;
  }

  // Weight tying: a Linear or an Embedding can share its weight matrix (out × in, or num × dim)
  // with another one of the same size, in the same module. params.tie = the id of the layer that owns it.
  const weightShape = n => (n && n.type === 'linear' ? [n.params.out, n.params.in] : n && n.type === 'embed' ? [n.params.num, n.params.dim] : null);
  function tieOwner(g, n, byId) {
    let m = n, k = 0; const get = id => (byId ? byId.get(id) : g.nodes.find(z => z.id === id));
    while (m && m.params.tie && k++ < 20) { const o = get(m.params.tie); if (!o || !weightShape(o)) return null; m = o; }
    return m === n ? null : m;
  }
  function makeNode(type, params = {}, x = 0, y = 0) {
    return { id: uid('n'), type, x, y, params: Object.assign({}, DEFAULTS[type], params) };
  }
  function inCount(n) {
    switch (n.type) {
      case 'input': return 0;
      case 'binop': return hasConst(n) ? 1 : 2;
      case 'func': return Math.max(1, +n.params.n || 1);
      case 'block': return Math.max(1, +n.params.nIn || 1);
      case 'loss': return 2;
      case 'rope': return n.params.mode === 'axial' || n.params.mode === 'mrope' ? 2 : 1;
      case 'mask': return n.params.mode === 'given' ? 2 : 1;
      default: return 1;
    }
  }
  const hasConst = n => n.params.const !== '' && n.params.const != null;
  const isCircle = n => n.type === 'binop';

  // ── graph helpers ──
  // An edge leaves its source's output e.out (0 unless the source is a block with several outputs).
  // Shapes and inputs refer to a source by key: the node id for output 0, "id@k" for output k.
  const srcKey = e => (e.out ? e.from + '@' + e.out : e.from);
  const splitKey = k => { const m = /^(.*)@(\d+)$/.exec(k); return m ? [m[1], +m[2]] : [k, 0]; };
  function index(g) {
    const byId = new Map(g.nodes.map(n => [n.id, n]));
    const ins = new Map(g.nodes.map(n => [n.id, new Array(inCount(n)).fill(null)]));
    const inKeys = new Map(g.nodes.map(n => [n.id, new Array(inCount(n)).fill(null)]));
    const outs = new Map(g.nodes.map(n => [n.id, []]));
    for (const e of g.edges) {
      if (!byId.has(e.from) || !byId.has(e.to)) continue;
      const arr = ins.get(e.to);
      if (e.port < arr.length) { arr[e.port] = e.from; inKeys.get(e.to)[e.port] = srcKey(e); }
      outs.get(e.from).push({ to: e.to, port: e.port, out: e.out || 0, edge: e });
    }
    const groupOf = new Map();
    for (const gr of g.groups || []) for (const id of gr.nodes) groupOf.set(id, gr);
    return { byId, ins, inKeys, outs, groupOf };
  }
  function topo(g, ix = index(g)) {
    const indeg = new Map(g.nodes.map(n => [n.id, 0]));
    for (const n of g.nodes) for (const s of ix.ins.get(n.id)) if (s) indeg.set(n.id, indeg.get(n.id) + 1);
    const q = g.nodes.filter(n => indeg.get(n.id) === 0).map(n => n.id);
    const order = [];
    while (q.length) {
      const id = q.shift(); order.push(id);
      for (const o of ix.outs.get(id)) { indeg.set(o.to, indeg.get(o.to) - 1); if (indeg.get(o.to) === 0) q.push(o.to); }
    }
    const cyclic = g.nodes.filter(n => !order.includes(n.id)).map(n => n.id);
    return { order, cyclic };
  }

  // ── dimension expressions ──
  const IDENT = /\b[A-Za-z_]\w*\b/g;
  function evalDim(expr, symbols) {
    if (expr === '' || expr == null) return null;
    const s = String(expr).trim();
    if (/^-?\d+(\.\d+)?$/.test(s)) return +s;
    if (!/^[\w\s+\-*/().]+$/.test(s)) return null;
    let missing = false;
    const js = s.replace(IDENT, id => {
      if (symbols && symbols[id] != null && symbols[id] !== '' && !isNaN(+symbols[id])) return '(' + (+symbols[id]) + ')';
      missing = true; return '0';
    }).replace(/\/\//g, '/');
    if (missing) return null;
    try { const v = Function('"use strict";return (' + js + ')')(); return Number.isFinite(v) ? Math.floor(v) : null; } catch { return null; }
  }
  function dimEq(a, b, symbols) {
    if (String(a).trim() === String(b).trim()) return true;
    const x = evalDim(a, symbols), y = evalDim(b, symbols);
    if (x != null && y != null) return x === y;
    return null;
  }
  const parseShape = s => String(s || '').split(',').map(t => t.trim()).filter(Boolean);
  const identsIn = expr => (String(expr).match(IDENT) || []);

  // ── shape inference ──
  function inferShapes(g, depth = 0) {
    const ix = index(g), { order, cyclic } = topo(g, ix);
    const shapes = new Map(), errors = new Map(), conflicts = new Map(), sym = g.symbols || {};
    // Size conflict at an operation: which inputs and which dims disagree.
    const conflict = (id, info) => { if (!conflicts.has(id)) conflicts.set(id, info); err(id, info.msg); };
    const show = d => String(d);
    // NumPy-style broadcasting, aligned from the right
    const broadcast = (a, b) => {
      const n = Math.max(a.length, b.length), out = []; let bad = null;
      for (let i = 1; i <= n; i++) {
        const x = a[a.length - i], y = b[b.length - i];
        if (x === undefined) out.unshift(y); else if (y === undefined) out.unshift(x);
        else if (x === '1') out.unshift(y); else if (y === '1') out.unshift(x);
        else { if (!bad && dimEq(x, y, sym) === false) bad = [x, y, a.length - i, b.length - i]; out.unshift(x); }
      }
      return { out, bad };
    };
    const err = (id, m) => { if (!errors.has(id)) errors.set(id, m); };
    for (const id of cyclic) err(id, 'Part of a cycle');
    for (const id of order) {
      const n = ix.byId.get(id), p = n.params;
      const inS = ix.inKeys.get(id).map(s => (s ? shapes.get(s) : undefined));
      if (n.type !== 'input' && inS.some(s => s === undefined)) {
        if (ix.ins.get(id).some(s => !s)) err(id, 'Unconnected input');
        shapes.set(id, null); continue;
      }
      if (inS.some(s => s === null)) { shapes.set(id, null); continue; }
      let out = null;
      const last = s => s[s.length - 1];
      switch (n.type) {
        case 'input': out = parseShape(p.shape); break;
        case 'output': out = inS[0]; break;
        case 'linear':
          if (dimEq(last(inS[0]), p.in, sym) === false) err(id, `Expects ${p.in}, got ${last(inS[0])}`);
          out = inS[0].slice(0, -1).concat(String(p.out)); break;
        case 'norm': {
          const x = inS[0], C = last(x);
          if (dimEq(C, p.dim, sym) === false) err(id, `Normalizes ${p.dim}, got ${C}`);
          if (p.kind === 'GroupNorm') {
            const c = evalDim(C, sym), gN = +p.groups || 8;
            if (c != null && c % gN !== 0) conflict(id, { kind: 'groups', a: C, b: String(gN), ports: [0], cap: `${C} (${c}) is not divisible by ${gN} groups`, msg: `GroupNorm needs the width to divide into ${gN} groups, but ${C} = ${c}` });
          }
          if (p.kind === 'InstanceNorm1d' && x.length < 3) conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × t × d', ports: [0], cap: 'needs a sequence dimension', msg: 'InstanceNorm normalizes over a sequence, so its input needs a shape like e × t × d' });
          if (NORMS[p.kind] && NORMS[p.kind].channelFirst && x.length > 3) conflict(id, { kind: 'rank', a: x.join('×'), b: 'rank ≤ 3', ports: [0], cap: 'input has more than 3 dims', msg: `${NORMS[p.kind].label} here supports inputs of shape e × d or e × t × d` });
          out = x; break;
        }
        case 'act': case 'dropout': case 'scale': out = inS[0]; break;
        case 'loss': {
          const [P, Y] = inS, fn = p.fn || 'cross_entropy', mode = fn === 'cross_entropy' ? p.mode || 'given' : 'given';
          const same = (a, b) => a.length === b.length && a.every((d, i) => dimEq(d, b[i], sym) !== false);
          const txt = x => x.map(show).join(' × ');
          if (fn === 'cross_entropy') {
            const want = P.slice(0, -1);
            if (mode === 'next' && P.length < 3) conflict(id, { kind: 'loss', a: txt(P), b: 'e × t × classes', ports: [0], cap: 'needs a sequence', msg: 'Next-token loss needs sequence logits (e × t × vocab)' });
            else if (!same(want, Y)) conflict(id, { kind: 'loss', a: txt(Y), b: txt(want), ports: [1], cap: `target should be ${txt(want)}`, msg: `Cross-entropy takes class indices shaped like the logits without the class dimension: the target is ${txt(Y)}, it should be ${txt(want)}` });
          } else if (!same(P, Y)) conflict(id, { kind: 'loss', a: txt(Y), b: txt(P), ports: [1], cap: `target should be ${txt(P)}`, msg: `${(LOSSES[fn] || {}).label || 'This loss'} compares two tensors of the same shape: the target is ${txt(Y)}, the output is ${txt(P)}` });
          out = []; break;
        }
        case 'rope': {
          const x = inS[0], D = evalDim(last(x), sym), P = inS[1];
          // positions: (T, A) for axial (A axes, each its share of the features), (A, B, T) for M-RoPE
          if (P && p.mode === 'axial' && x.length >= 2 && P.length === 2 && dimEq(P[0], x[x.length - 2], sym) === false) conflict(id, { kind: 'rope', a: P[0], b: x[x.length - 2], ports: [1], cap: 'positions ≠ tokens', msg: `There are ${P[0]} positions for ${x[x.length - 2]} tokens` });
          if (P && p.mode === 'mrope' && x.length >= 2 && P.length >= 2 && dimEq(last(P), x[x.length - 2], sym) === false) conflict(id, { kind: 'rope', a: last(P), b: x[x.length - 2], ports: [1], cap: 'positions ≠ tokens', msg: `There are ${last(P)} positions for ${x[x.length - 2]} tokens` });
          if (p.mode === 'mrope' && D != null) { const tot = String(p.sections || '').split(',').reduce((a, v) => a + (+v || 0), 0); if (tot * 2 !== D) conflict(id, { kind: 'rope', a: String(tot * 2), b: String(D), ports: [0], cap: 'sections ≠ width', msg: `The M-RoPE sections cover ${tot} feature pairs, but the width is ${D} (${D / 2} pairs)` }); }
          if (x.length < 2) conflict(id, { kind: 'rank', a: x.join('×'), b: '… × t × d', ports: [0], cap: 'needs positions', msg: 'Rotary embeddings need a sequence input like e × h × t × d' });
          else if (D != null && D % 2) conflict(id, { kind: 'rope', a: last(x), b: 'even', ports: [0], cap: 'odd width', msg: `Rotary embeddings rotate pairs of features, so the width must be even (it is ${D})` });
          out = x; break;
        }
        case 'embed': {
          const x = inS[0];
          // positions: embeds 0…T−1 where T is the input's sequence length, giving (T, dim) that broadcasts over the batch
          out = p.positions ? [x[x.length - 1], String(p.dim)] : x.concat(String(p.dim));
          if (p.positions && x.length < 2) err(id, 'Position embeddings need a sequence input like e × t');
          break;
        }
        case 'heads': {
          const x = inS[0], H = String(p.heads), C = last(x);
          if (x.length < 2) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × t × d', ports: [0], cap: 'needs e × t × d', msg: 'Splitting into heads needs a sequence input like e × t × d' }); out = x; break; }
          const c = evalDim(C, sym), h = evalDim(H, sym);
          if (c != null && h != null && (h <= 0 || c % h !== 0)) conflict(id, { kind: 'heads', a: C, b: H, ports: [0], msg: `The width ${C} doesn't divide into ${H} heads` });
          out = x.slice(0, -2).concat(H, x[x.length - 2], /^[A-Za-z_]\w*$|^\d+$/.test(C) ? `${C}/${H}` : `(${C})/${H}`);
          break;
        }
        case 'merge': {
          const x = inS[0];
          if (x.length < 3) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × h × t × d', ports: [0], cap: 'needs heads', msg: 'Merging heads needs an input like e × h × t × d' }); out = x; break; }
          const H = x[x.length - 3], D = last(x), m = new RegExp('^\\(?(.+?)\\)?/' + String(H).replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$').exec(D);
          out = x.slice(0, -3).concat(x[x.length - 2], m ? m[1] : `${H}*${D}`);
          break;
        }
        case 'transpose': { const x = inS[0]; out = x.length < 2 ? x : x.slice(0, -2).concat(last(x), x[x.length - 2]); break; }
        case 'mask': {
          const x = inS[0];
          if (p.mode === 'given') {
            const M = inS[1];
            if (M && M.length >= 2 && x.length >= 2 && [1, 2].some(k => dimEq(M[M.length - k], x[x.length - k], sym) === false)) conflict(id, { kind: 'mask', a: M.slice(-2).join('×'), b: x.slice(-2).join('×'), ports: [1], cap: 'mask ≠ scores', msg: `The mask is ${M.slice(-2).join(' × ')} but the scores are ${x.slice(-2).join(' × ')}` });
            out = x; break;
          }
          if (x.length < 2 || dimEq(x[x.length - 2], last(x), sym) === false) conflict(id, { kind: 'mask', a: x[x.length - 2] || '?', b: last(x), ports: [0], msg: 'A causal mask needs square scores (… × t × t)' });
          out = x; break;
        }
        case 'binop': {
          if (inS.length === 2) {
            const [a, b] = inS;
            if (p.op === '@') { out = matmul(id, a, b); break; }
            const r = broadcast(a, b); out = r.out;
            if (r.bad) conflict(id, { kind: 'elementwise', op: p.op, a: r.bad[0], b: r.bad[1], ports: [0, 1], msg: `Sizes disagree for ${p.op === '*' ? '⊙' : p.op}: ${show(r.bad[0])} vs ${show(r.bad[1])}` });
          } else out = inS[0];
          break;
        }
        case 'block': {
          const blk = g.blocks && g.blocks[p.block];
          if (!blk) { err(id, 'This block no longer exists'); out = null; break; }
          if (depth > 24) { err(id, `${blk.name} contains itself`); out = null; break; }
          const inner = instantiateBlock(g, n, inS);
          const r = inferShapes(inner, depth + 1);
          // every output of the block: the first is the node's own shape, output k is "id@k"
          const outsS = inner.nodes.filter(z => z.type === 'output').map(on => { const oe = inner.edges.find(z => z.to === on.id); return oe ? r.shapes.get(srcKey(oe)) || null : null; });
          out = outsS[0] || null;
          outsS.forEach((sh, k) => { if (k) shapes.set(id + '@' + k, sh); });
          if (r.conflicts.size) {
            const c = [...r.conflicts.values()][0];
            conflict(id, { kind: 'inner', where: blk.name, a: c.a, b: c.b, inner: c, ports: [], msg: `Inside ${blk.name}: ${c.msg}` });
          } else if (!out || outsS.some(x => !x)) err(id, `${blk.name} has an output that isn't connected`);
          // repeating needs the output to have exactly the input's shape (each copy feeds the next)
          const reps = Math.max(1, Math.floor(+p.repeat || 1));
          if (reps > 1 && out && !conflicts.has(id)) {
            const x = inS[0];
            // copies chain through the first input; any other inputs go to every copy
            if (outsS.length > 1) conflict(id, { kind: 'repeat', a: String(outsS.length), b: '1', ports: [], cap: 'repeat needs one output', msg: `${blk.name} can only repeat with a single output` });
            else if (x.length !== out.length || x.some((d, i) => dimEq(d, out[i], sym) === false)) conflict(id, { kind: 'repeat', a: last(x), b: last(out), ports: [0], cap: `can't repeat: ${x.join(' × ')} in, ${out.join(' × ')} out`, msg: `Repeating ${blk.name} needs its output to match its input (${x.join(' × ')} in, ${out.join(' × ')} out)` });
          }
          break;
        }
        case 'func': {
          const m = String(p.extra || '').match(/dim\s*=\s*(-?\d+)/);
          const fn = String(p.fn || '');
          const ex = String(p.extra || '').trim(), x0 = inS[0], ax = k => (k < 0 ? x0.length + k : k), par = d => (/^[A-Za-z_]\w*$|^\d+$/.test(d) ? d : `(${d})`);
          if (HELPER_OPS[fn]) { out = HELPER_OPS[fn].shape(inS, sym); break; }
          // x.unsqueeze(k) / x.squeeze(k): a size-1 dimension in or out
          if (/(^|\.)unsqueeze$/.test(fn) && /^-?\d+$/.test(ex)) { const k = +ex < 0 ? x0.length + 1 + +ex : +ex; out = x0.slice(0, k).concat('1', x0.slice(k)); break; }
          if (/(^|\.)squeeze$/.test(fn) && /^-?\d+$/.test(ex)) { const k = ax(+ex); out = String(x0[k]) === '1' ? x0.slice(0, k).concat(x0.slice(k + 1)) : x0; break; }
          // x.repeat_interleave(r, dim=k): that size times r (e.g. key/value heads shared by several query heads)
          if (/(^|\.)repeat_interleave$/.test(fn) && inS.length === 1) {
            const mm = /^([A-Za-z_]\w*|\d+)\s*,\s*(?:dim\s*=\s*)?(-?\d+)$/.exec(ex);
            if (mm) { const k = ax(+mm[2]); out = x0.map((d, i) => (i === k ? `${par(d)}*${mm[1]}` : d)); break; }
          }
          // x.unflatten(k, (a, b)): size k becomes a × b (-1 is worked out)
          if (/(^|\.)unflatten$/.test(fn) && inS.length === 1) {
            const mm = /^(-?\d+)\s*,\s*\(\s*(-1|[A-Za-z_]\w*|\d+)\s*,\s*(-1|[A-Za-z_]\w*|\d+)\s*\)$/.exec(ex);
            if (mm) { const k = ax(+mm[1]), D = x0[k]; const A = mm[2] === '-1' ? `${par(D)}/${mm[3]}` : mm[2], B = mm[3] === '-1' ? `${par(D)}/${mm[2]}` : mm[3]; out = x0.slice(0, k).concat(A, B, x0.slice(k + 1)); break; }
          }
          // x.flatten(a[, b]): sizes a … b (default: to the end) multiply into one
          if (/(^|\.)flatten$/.test(fn) && inS.length === 1) {
            const mm = /^(-?\d+)(?:\s*,\s*(-?\d+))?$/.exec(ex);
            if (mm) { const s0 = ax(+mm[1]), s1 = mm[2] != null ? ax(+mm[2]) : x0.length - 1; out = x0.slice(0, s0).concat(x0.slice(s0, s1 + 1).map(par).join('*'), x0.slice(s1 + 1)); break; }
          }
          if (/(^|\.)(matmul|mm|bmm)$/.test(fn) && inS.length === 2) { out = matmul(id, inS[0], inS[1]); break; }
          if (/(^|\.)(cat|concat|concatenate)$/.test(fn) && inS.length > 1) {
            const ax = m ? +m[1] : 0, r0 = inS[0], k = ax < 0 ? r0.length + ax : ax;
            for (let j = 1; j < inS.length && !conflicts.has(id); j++) {
              const r1 = inS[j];
              if (r1.length !== r0.length) { conflict(id, { kind: 'cat', a: r0.join('×'), b: r1.join('×'), ports: [0, j], msg: `Concatenated tensors have different ranks: ${r0.join(' × ')} vs ${r1.join(' × ')}` }); break; }
              r0.forEach((d, i) => { if (i !== k && !conflicts.has(id) && dimEq(d, r1[i], sym) === false) conflict(id, { kind: 'cat', a: d, b: r1[i], ports: [0, j], msg: `Sizes disagree outside the concat axis: ${show(d)} vs ${show(r1[i])}` }); });
            }
            out = r0.map((d, i) => (i === k ? inS.map(s => s[i]).join(' + ') : d));
            break;
          }
          if (/(^|\.)stack$/.test(fn) && inS.length > 1) {
            for (let j = 1; j < inS.length && !conflicts.has(id); j++) { const r = broadcast(inS[0], inS[j]); if (r.bad || inS[j].length !== inS[0].length) conflict(id, { kind: 'stack', a: r.bad ? r.bad[0] : inS[0].join('×'), b: r.bad ? r.bad[1] : inS[j].join('×'), ports: [0, j], msg: 'Stacked tensors must have the same shape' }); }
            out = [String(inS.length), ...inS[0]]; break;
          }
          out = inS[0];
          break;
        }
      }
      if ((n.type === 'linear' || n.type === 'embed') && p.tie) {
        const o = ix.byId.get(p.tie), ws = weightShape(n), wo = weightShape(o);
        if (!wo) err(id, 'Its weights are tied to a layer that no longer exists');
        else if (ws.some((d, i) => dimEq(d, wo[i], sym) === false)) {
          const txt = w => w.map(show).join(' × ');
          conflict(id, { kind: 'tie', a: txt(ws), b: txt(wo), ports: [], cap: 'tied weights differ in size', msg: `Tied weights need the same size: ${txt(ws)} here, ${txt(wo)} in the layer it shares with` });
        }
      }
      shapes.set(id, out);
    }
    return { shapes, errors, conflicts };
    // a @ b: inner sizes must agree; leading (batch) dims broadcast
    function matmul(id, a, b) {
      if (b.length === 1) {
        if (dimEq(last(a), b[0], sym) === false) conflict(id, { kind: 'matmul', op: '@', a: last(a), b: b[0], ports: [0, 1], msg: `Matrix multiply needs matching inner sizes: ${show(last(a))} vs ${show(b[0])}` });
        return a.slice(0, -1);
      }
      if (a.length === 1) {
        if (dimEq(a[0], b[b.length - 2], sym) === false) conflict(id, { kind: 'matmul', op: '@', a: a[0], b: b[b.length - 2], ports: [0, 1], msg: `Matrix multiply needs matching inner sizes: ${show(a[0])} vs ${show(b[b.length - 2])}` });
        return b.slice(0, -2).concat(last(b));
      }
      const inner = [last(a), b[b.length - 2]];
      if (dimEq(inner[0], inner[1], sym) === false) conflict(id, { kind: 'matmul', op: '@', a: inner[0], b: inner[1], ports: [0, 1], msg: `Matrix multiply needs the left input's last size to equal the right input's second-to-last: ${show(inner[0])} vs ${show(inner[1])}` });
      const bat = broadcast(a.slice(0, -2), b.slice(0, -2));
      return bat.out.concat(a[a.length - 2], last(b));
    }
    function last(s) { return s[s.length - 1]; }
  }
  function paramCount(n, symbols) {
    const p = n.params, d = x => evalDim(x, symbols);
    if (n.type === 'linear') { const i = d(p.in), o = d(p.out); return i == null || o == null ? null : (p.tie ? 0 : i * o) + (p.bias ? o : 0); }
    if (n.type === 'embed') { const a = d(p.num), b = d(p.dim); return a == null || b == null ? null : p.tie ? 0 : a * b; }
    if (n.type === 'norm') { const k = d(p.dim); return k == null ? null : ({ LayerNorm: 2 * k, BatchNorm1d: 2 * k, GroupNorm: 2 * k, InstanceNorm1d: 0 }[p.kind] ?? k); }
    return 0;
  }

  // A Linear's input width and a Norm's width always follow the tensor flowing in,
  // so sizes are only ever set where they are created (Linear outputs, inputs).
  function syncWidths(g) {
    const { shapes } = inferShapes(g), ix = index(g); let changed = 0;
    for (const n of g.nodes) {
      if (n.type !== 'linear' && n.type !== 'norm') continue;
      const src = ix.ins.get(n.id)[0], sh = src && shapes.get(src);
      if (!sh || !sh.length) continue;
      const d = String(sh[sh.length - 1]), f = n.type === 'linear' ? 'in' : 'dim', cur = String(n.params[f]);
      // a layer keeps its own plain size when what arrives is a formula worth the same (heads × head width = d)
      if (cur !== d && /^[A-Za-z_]\w*$|^\d+$/.test(cur) && !/^[A-Za-z_]\w*$|^\d+$/.test(d) && dimEq(cur, d, g.symbols) === true) continue;
      if (cur !== d) { n.params[f] = d; changed++; }
    }
    return changed;
  }

  // Link sizes that must be equal (both inputs of an elementwise op) so they share one
  // named width: editing one then updates the other. Used after importing literal code.
  function linkSizes(g) {
    const { shapes } = inferShapes(g), ix = index(g);
    const parent = new Map(), find = k => { while (parent.get(k) !== k) { parent.set(k, parent.get(parent.get(k))); k = parent.get(k); } return k; };
    const add = k => { if (!parent.has(k)) parent.set(k, k); return k; };
    const union = (a, b) => { a = find(add(a)); b = find(add(b)); if (a !== b) parent.set(b, a); };
    // slot = the place a tensor's last size is set: a Linear's out, or an input's last dim
    const slotOf = id => {
      let n = ix.byId.get(id), guard = 0;
      while (n && guard++ < 500) {
        if (n.type === 'linear') return n.id + '|out';
        if (n.type === 'embed') return n.id + '|dim';
        if (n.type === 'input') return n.id + '|' + (parseShape(n.params.shape).length - 1);
        if (n.type === 'func' && n.params.init) return null;
        const src = ix.ins.get(n.id)[0]; n = src ? ix.byId.get(src) : null;
      }
      return null;
    };
    const get = k => { const [id, f] = k.split('|'), n = ix.byId.get(id); return isNaN(+f) ? String(n.params[f]) : parseShape(n.params.shape)[+f]; };
    const put = (k, v) => { const [id, f] = k.split('|'), n = ix.byId.get(id); if (isNaN(+f)) n.params[f] = v; else { const d = parseShape(n.params.shape); d[+f] = v; n.params.shape = d.join(', '); } };
    // sizes written with the same name are one size, even where no elementwise op joins them
    const byName = new Map();
    for (const n of g.nodes) {
      const k = n.type === 'linear' ? n.id + '|out' : n.type === 'embed' ? n.id + '|dim' : n.type === 'input' ? n.id + '|' + (parseShape(n.params.shape).length - 1) : null;
      if (!k) continue; const v = get(k);
      if (/^[A-Za-z_]\w*$/.test(v)) { if (byName.has(v)) union(byName.get(v), k); else byName.set(v, add(k)); }
    }
    for (const n of g.nodes) {
      if (n.type !== 'binop' || n.params.op === '@' || hasConst(n)) continue;
      const [a, b] = ix.ins.get(n.id); if (!a || !b) continue;
      const sa = shapes.get(a), sb = shapes.get(b); if (!sa || !sb) continue;
      if (dimEq(sa[sa.length - 1], sb[sb.length - 1], g.symbols) !== true) continue;
      const ka = slotOf(a), kb = slotOf(b); if (ka && kb) union(ka, kb);
    }
    const classes = new Map();
    for (const k of parent.keys()) { const r = find(k); if (!classes.has(r)) classes.set(r, []); classes.get(r).push(k); }
    let linked = 0;
    for (const ks of classes.values()) {
      if (ks.length < 2) continue;
      const vals = ks.map(get);
      if (new Set(vals).size === 1 && /^[A-Za-z_]\w*$/.test(vals[0])) continue;
      let rep = vals.find(v => /^[A-Za-z_]\w*$/.test(v));
      if (!rep) { const v = evalDim(vals[0], g.symbols); let nm = 'd' + v, k = 2; while (g.symbols[nm] != null && +g.symbols[nm] !== v) nm = 'd' + v + '_' + k++; g.symbols[nm] = v; rep = nm; }
      ks.forEach(k => put(k, rep)); linked++;
    }
    syncWidths(g);
    return linked;
  }

  // Rename a tensor width produced by a node and carry the change downstream
  // through width-preserving ops until the next layer that consumes it.
  function propagateWidth(g, nodeId, oldDim, newDim) {
    if (String(oldDim) === String(newDim)) return [];
    const ix = index(g), changed = [], seen = new Set();
    const walk = id => {
      for (const o of ix.outs.get(id)) {
        const k = o.to + ':' + o.port; if (seen.has(k)) continue; seen.add(k);
        const n = ix.byId.get(o.to), p = n.params;
        if (n.type === 'linear') { if (String(p.in) === String(oldDim)) { p.in = newDim; changed.push(n.id); } }
        else if (n.type === 'norm') { if (String(p.dim) === String(oldDim)) { p.dim = newDim; changed.push(n.id); } walk(n.id); }
        else if (n.type === 'act' || n.type === 'dropout' || n.type === 'binop' || n.type === 'output') walk(n.id);
      }
    };
    walk(nodeId);
    return changed;
  }

  // ── code generation ──
  const ACT_CODE = {
    sigmoid: x => `torch.sigmoid(${x})`, silu: x => `F.silu(${x})`, relu: x => `F.relu(${x})`,
    gelu: x => `F.gelu(${x})`, gelu_tanh: x => `F.gelu(${x}, approximate="tanh")`, tanh: x => `torch.tanh(${x})`,
  };
  const OP_PREC = { '+': 10, '-': 10, '*': 20, '/': 20, '@': 20 };
  const DEF_ATTR = { linear: 'fc', norm: 'norm', dropout: 'drop', func: 'mod', embed: 'emb' };
  const DEF_VAR = n => n.params.var || ({ linear: 'h', norm: 'n', act: 'a', dropout: 'h', func: 'y', embed: 'x', heads: 'h', merge: 'y', transpose: 't', scale: 's', mask: 'att', rope: 'r' }[n.type] || (n.params.op === '*' ? 'g' : 'x'));
  const DIM_FIELDS = { linear: ['in', 'out'], norm: ['dim'], embed: ['num', 'dim'], heads: ['heads'] };
  // [field, expression] for every size a node is built with; a block use's sizes are its args
  const dimList = n => (DIM_FIELDS[n.type] || []).map(f => [f, n.params[f]]).concat(n.type === 'block' ? Object.entries(n.params.args || {}).sort().map(([k, v]) => ['args.' + k, v]) : []);
  const baseOf = k => splitKey(String(k))[0];
  // small helper functions written above the classes when the model uses them
  const HELPERS = {
    heads: `def split_heads(x, n_head):\n    \"\"\"(B, T, C) -> (B, n_head, T, C // n_head)\"\"\"\n    return x.unflatten(-1, (n_head, -1)).transpose(1, 2)`,
    merge: `def merge_heads(x):\n    \"\"\"(B, n_head, T, D) -> (B, T, n_head * D)\"\"\"\n    return x.transpose(1, 2).flatten(2)`,
    rope: `def apply_rope(x, base=10000.0):\n    \"\"\"Rotary position embedding on (B, n_head, T, D): rotates feature pairs (i, i + D/2) by position-dependent angles.\"\"\"\n    t, d = x.size(-2), x.size(-1)\n    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)\n    ang = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), inv_freq)\n    ang = torch.cat((ang, ang), dim=-1)\n    x1, x2 = x[..., : d // 2], x[..., d // 2 :]\n    return x * ang.cos().to(x.dtype) + torch.cat((-x2, x1), dim=-1) * ang.sin().to(x.dtype)`,
    mask: `def causal_mask(scores):\n    \"\"\"Hide future positions: scores above the diagonal become -inf.\"\"\"\n    t = scores.size(-1)\n    future = torch.ones(t, t, dtype=torch.bool, device=scores.device).triu(1)\n    return scores.masked_fill(future, float("-inf"))`,
  };

  // More helpers, written above the classes when used: rotary embeddings at given positions, a mask given as an
  // input, and named ops (custom ops whose function is one of these) for vision-language models like Qwen2.5-VL.
  Object.assign(HELPERS, {
    ropeAxial: String.raw`def apply_rope_axial(x, pos, base=10000.0):
    """Rotary embedding at given positions with several axes, e.g. an image patch's (row, column): pos is (T, A) and
    each axis rotates its own share of the features of x (B, n_head, T, D)."""
    d, a = x.size(-1), pos.size(-1)
    inv_freq = 1.0 / base ** (torch.arange(0, d // a, 2, device=x.device, dtype=torch.float32) / (d // a))
    ang = (pos[..., None].float() * inv_freq).flatten(-2)
    ang = torch.cat((ang, ang), dim=-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * ang.cos().to(x.dtype) + torch.cat((-x2, x1), dim=-1) * ang.sin().to(x.dtype)`,
    ropeMrope: String.raw`def apply_rope_mrope(x, pos, base=10000.0, sections=(16, 24, 24)):
    """Multimodal rotary embedding (M-RoPE): pos (A, B, T) gives every token one position per axis (time, row,
    column; all equal for text). The rotated feature pairs are cut into sections; section i uses axis i % A."""
    d = x.size(-1)
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos[..., None].float() * inv_freq
    ang = torch.cat([part[i % pos.size(0)] for i, part in enumerate(ang.split(list(sections), dim=-1))], dim=-1)
    ang = torch.cat((ang, ang), dim=-1)[:, None]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * ang.cos().to(x.dtype) + torch.cat((-x2, x1), dim=-1) * ang.sin().to(x.dtype)`,
    maskGiven: String.raw`def apply_mask(scores, allowed):
    """Attention only where allowed (boolean, broadcast against the scores); everywhere else -inf."""
    return scores.masked_fill(~allowed, float("-inf"))`,
  });
  const par0 = d => (/^[A-Za-z_]\w*$|^\d+$/.test(String(d)) ? d : `(${d})`);
  // name → { code, deps (other named ops it calls), shape(inS, sym) }
  const HELPER_OPS = {
    vision_window_index: { defaults: { n: 2, extra: 'merge=2, window=112, patch=14' }, deps: [], shape: inS => [`${par0(inS[0][0])}/4`], code: String.raw`def vision_window_index(x, grid, merge=2, window=112, patch=14):
    """Qwen2.5-VL's window order: the merge x merge patch groups of every image, listed window by window (a window
    covers window x window pixels). x is the patches (only its device is used), grid the (t, h, w) of each image."""
    index, start, win = [], 0, window // merge // patch
    for t, h, w in grid.tolist():
        gh, gw = h // merge, w // merge
        idx = torch.arange(t * gh * gw).reshape(t, gh, gw)
        ph, pw = win - gh % win, win - gw % win
        idx = F.pad(idx, (0, pw, 0, ph), value=-100).reshape(t, (gh + ph) // win, win, (gw + pw) // win, win)
        idx = idx.permute(0, 1, 3, 2, 4).reshape(-1)
        index.append(idx[idx != -100] + start)
        start += t * gh * gw
    return torch.cat(index).to(x.device)` },
    window_order: { defaults: { n: 2, extra: '' }, deps: [], shape: inS => inS[0], code: String.raw`def window_order(x, index, group=4):
    """Reorder the rows of x in groups of group rows (merge x merge patches), group i taking group index[i]."""
    return x.reshape(-1, group, *x.shape[1:])[index].reshape(x.shape)` },
    window_restore: { defaults: { n: 2, extra: '' }, deps: [], shape: inS => inS[0], code: String.raw`def window_restore(x, index):
    """Undo window_order for one row per group (the merged patches)."""
    return x[torch.argsort(index)]` },
    vision_positions: { defaults: { n: 2, extra: 'merge=2' }, deps: [], shape: inS => [inS[0][0], '2'], code: String.raw`def vision_positions(x, grid, merge=2):
    """(row, column) of every patch, in the order the image processor lists them (merge x merge groups)."""
    pos = []
    for t, h, w in grid.tolist():
        hp, wp = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        blocks = (h // merge, merge, w // merge, merge)
        hw = torch.stack([hp.reshape(blocks).transpose(1, 2).flatten(), wp.reshape(blocks).transpose(1, 2).flatten()], dim=-1)
        pos.append(hw.repeat(t, 1))
    return torch.cat(pos).to(x.device)` },
    vision_window_mask: { defaults: { n: 2, extra: 'merge=2, window=112, patch=14' }, deps: ['vision_window_index'], shape: inS => [inS[0][0], inS[0][0]], code: String.raw`def vision_window_mask(x, grid, merge=2, window=112, patch=14):
    """Which patches may attend to which, in window order: those in the same window of the same frame."""
    win, seg, start = window // merge // patch, [], 0
    for t, h, w in grid.tolist():
        gh, gw = h // merge, w // merge
        nh, nw = (gh + win - gh % win) // win, (gw + win - gw % win) // win
        cell = (torch.arange(gh)[:, None] // win) * nw + torch.arange(gw)[None, :] // win
        seg.append((torch.arange(t)[:, None, None] * (nh * nw) + cell).reshape(-1) + start)
        start += t * nh * nw
    seg = torch.cat(seg).to(x.device)[vision_window_index(x, grid, merge, window, patch)].repeat_interleave(merge * merge)
    return seg[:, None] == seg[None, :]` },
    vision_frame_mask: { defaults: { n: 2, extra: 'merge=2, window=112, patch=14' }, deps: ['vision_window_index'], shape: inS => [inS[0][0], inS[0][0]], code: String.raw`def vision_frame_mask(x, grid, merge=2, window=112, patch=14):
    """Which patches may attend to which, in window order: those in the same frame of the same image."""
    seg, start = [], 0
    for t, h, w in grid.tolist():
        seg.append(torch.arange(t).repeat_interleave((h // merge) * (w // merge)) + start)
        start += t
    seg = torch.cat(seg).to(x.device)[vision_window_index(x, grid, merge, window, patch)].repeat_interleave(merge * merge)
    return seg[:, None] == seg[None, :]` },
    place_tokens: { defaults: { n: 3, extra: 'token_id=151655' }, deps: [], shape: inS => inS[0], code: String.raw`def place_tokens(embeds, ids, features, token_id=151655):
    """Put one row of features into each placeholder token (ids == token_id), in order."""
    return embeds.masked_scatter((ids == token_id).unsqueeze(-1), features.to(embeds.dtype))` },
    mrope_positions: { defaults: { n: 2, extra: 'token_id=151655, merge=2' }, deps: [], shape: inS => ['3', ...inS[0]], code: String.raw`def mrope_positions(ids, grid, token_id=151655, merge=2):
    """(time, row, column) position of every token for M-RoPE (Qwen2.5-VL's rope index, images only): text counts
    up 0, 1, 2, ... on all three axes; a run of image tokens gets the frame, row and column of each merged patch,
    offset by the position reached so far, which then moves on by the image's larger side."""
    pos = torch.zeros(3, *ids.shape, dtype=torch.long, device=ids.device)
    grids = iter(grid.tolist())
    for b, row in enumerate(ids.tolist()):
        parts, cur, i = [], 0, 0
        while i < len(row):
            j, image = i, row[i] == token_id
            while j < len(row) and (row[j] == token_id) == image:
                j += 1
            if image:
                t, h, w = next(grids)
                tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h // merge), torch.arange(w // merge), indexing="ij")
                parts.append(torch.stack([tt, hh, ww]).reshape(3, -1) + cur)
                cur += max(h, w) // merge
            else:
                parts.append(torch.arange(cur, cur + j - i).expand(3, -1))
                cur += j - i
            i = j
        pos[:, b] = torch.cat(parts, dim=1).to(ids.device)
    return pos` },
  };
  const helperDoc = name => { const m = /"""([\s\S]*?)"""/.exec(HELPER_OPS[name].code); return m ? m[1].replace(/\s+/g, ' ').trim() : ''; };
  // which helper functions a set of nodes needs, in a fixed order (named ops after the ones they call)
  function helpersFor(nodes) {
    const want = new Set();
    for (const n of nodes) {
      if (n.type === 'heads') want.add('heads');
      if (n.type === 'merge') want.add('merge');
      if (n.type === 'rope') want.add(n.params.mode === 'axial' ? 'ropeAxial' : n.params.mode === 'mrope' ? 'ropeMrope' : 'rope');
      if (n.type === 'mask') want.add(n.params.mode === 'given' ? 'maskGiven' : 'mask');
      if (n.type === 'func' && HELPER_OPS[n.params.fn]) { want.add(n.params.fn); HELPER_OPS[n.params.fn].deps.forEach(d => want.add(d)); }
    }
    const order = ['heads', 'merge', 'rope', 'ropeAxial', 'ropeMrope', 'mask', 'maskGiven', ...Object.keys(HELPER_OPS)];
    return order.filter(k => want.has(k)).map(k => HELPERS[k] || HELPER_OPS[k].code);
  }

  function analyzeGroups(g, ix) {
    const out = [];
    for (const gr of g.groups || []) {
      const mem = new Set(gr.nodes.filter(id => ix.byId.has(id)));
      const info = { group: gr, members: mem, valid: true, why: '' };
      if (!mem.size) { info.valid = false; info.why = 'empty'; out.push(info); continue; }
      const outsExt = [...mem].filter(id => ix.outs.get(id).some(o => !mem.has(o.to)));
      const given = Array.isArray(gr.outs) && gr.outs.length && gr.outs.every(Boolean);
      if ([...mem].some(id => ['input', 'output'].includes(ix.byId.get(id).type))) { info.valid = false; info.why = 'contains an input or output node'; }
      else if (given ? outsExt.some(id => !gr.outs.some(o => baseOf(o) === id)) : outsExt.length !== 1) { info.valid = false; info.why = given ? 'feeds the model from a layer that is not one of its outputs' : outsExt.length ? 'has more than one output' : 'has no output'; }
      if (info.valid) {
        info.outputs = given ? gr.outs.slice() : [outsExt[0]];
        info.output = info.outputs[0];
        // canonical DFS from the outputs: fixes member order; external inputs keep the given order when there is one
        const order = [], extIn = given && Array.isArray(gr.extIn) ? [...new Set(gr.extIn.filter(Boolean))] : [], seen = new Set();
        const dfs = id => {
          if (seen.has(id)) return; seen.add(id);
          for (const s of ix.inKeys.get(id)) {
            if (!s) continue;
            if (mem.has(baseOf(s))) dfs(baseOf(s)); else if (!extIn.includes(s)) extIn.push(s);
          }
          order.push(id);
        };
        info.outputs.forEach(o => { if (mem.has(baseOf(o))) dfs(baseOf(o)); });
        info.order = order; info.extIn = extIn;
        if (given && Array.isArray(gr.inNames) && gr.extIn && new Set(gr.extIn).size === gr.extIn.length && gr.inNames.length === extIn.length) info.inNames = gr.inNames.slice();
        if (given && Array.isArray(gr.outNames)) info.outNames = gr.outNames.slice();
        if (order.length !== mem.size) { info.valid = false; info.why = 'has nodes that do not reach its output'; }
        else {
          // structural signature with dims abstracted
          const localIdx = new Map(order.map((id, i) => [id, i]));
          const ids = [];
          const dimSlots = [];
          for (const id of order) for (const [, v] of dimList(ix.byId.get(id))) {
            const e = String(v); dimSlots.push(e);
            for (const t of identsIn(e)) if (!ids.includes(t)) ids.push(t);
          }
          const pat = e => String(e).replace(IDENT, t => '$' + ids.indexOf(t));
          info.ids = ids;
          // the same structure with sizes left out: uses of one module class whatever their sizes are called
          const refK = s => { if (!s) return '-'; const [b, k] = splitKey(s); return mem.has(b) ? 'm' + localIdx.get(b) + (k ? '@' + k : '') : 'x' + extIn.indexOf(s); };
          info.ssig = order.map(id => { const n = ix.byId.get(id); return n.type + (n.type === 'block' ? ':' + n.params.block : '') + '|' + ix.inKeys.get(id).map(refK).join(','); }).join(';')
            + '#in' + extIn.length + '#out' + info.outputs.length;
          info.sig = order.map(id => {
            const n = ix.byId.get(id), p = n.params;
            const ins = ix.inKeys.get(id).map(refK);
            // a tie is compared by which member it points at, so copies of one block still share one class
            const keep = Object.keys(p).filter(k => !['name', 'title', 'tone', 'var', 'args', ...(DIM_FIELDS[n.type] || [])].includes(k)).sort().map(k => k + '=' + (k === 'tie' && p[k] ? (localIdx.has(p[k]) ? 'm' + localIdx.get(p[k]) : 'out') : p[k]));
            const dims = dimList(n).map(([, v]) => pat(v));
            return [n.type, keep.join(','), dims.join(','), ins.join(',')].join('|');
          }).join(';') + '#in' + extIn.length + '#out:' + info.outputs.map(refK).join(',');
        }
      }
      out.push(info);
    }
    return out;
  }

  // ── custom blocks ──
  // A block is a reusable sub-network {id, name, graph, labels} kept in g.blocks. Its graph has input node(s) and
  // one output node. A 'block' node in the diagram uses it: inputs come from the stream, and each size used inside
  // that isn't set by an input (e.g. a hidden width) is a per-use argument stored in node.params.args.
  const DIMF = { linear: ['in', 'out'], norm: ['dim'], func: ['init'], input: ['shape'], embed: ['num', 'dim'], heads: ['heads'] };
  function blockInputs(blk) { return blk.graph.nodes.filter(n => n.type === 'input'); }
  function blockOutputs(blk) { return blk.graph.nodes.filter(n => n.type === 'output'); }
  function blockParams(blk) {
    const fromInputs = new Set();
    for (const n of blockInputs(blk)) parseShape(n.params.shape).forEach(d => identsIn(d).forEach(t => fromInputs.add(t)));
    const ps = [];
    const exprs = n => (n.type === 'linear' ? [n.params.out] : n.type === 'func' && n.params.init ? [n.params.init] : n.type === 'heads' ? [n.params.heads] : n.type === 'embed' ? [n.params.num, n.params.dim] : n.type === 'block' ? Object.values(n.params.args || {}).map(String) : []);
    for (const n of blk.graph.nodes) for (const e of exprs(n))
      identsIn(e).forEach(t => { if (!fromInputs.has(t) && t !== BATCH && !ps.includes(t)) ps.push(t); });
    return ps;
  }
  // inner symbol → outer expression for one use
  function bindBlock(g, n, inS) {
    const blk = g.blocks[n.params.block], map = new Map();
    blockInputs(blk).forEach((inp, k) => {
      const dims = parseShape(inp.params.shape), sh = inS && inS[k]; if (!sh) return;
      for (let i = 1; i <= dims.length; i++) { const d = dims[dims.length - i], o = sh[sh.length - i]; if (o !== undefined && /^[A-Za-z_]\w*$/.test(d) && !map.has(d)) map.set(d, String(o)); }
    });
    for (const p of blockParams(blk)) if (!map.has(p)) {
      const a = n.params.args && n.params.args[p];
      map.set(p, a != null && a !== '' ? String(a) : String(blk.graph.symbols && blk.graph.symbols[p] != null ? blk.graph.symbols[p] : 64));
    }
    return map;
  }
  const substDims = (node, map) => {
    for (const f of DIMF[node.type] || []) if (f !== 'shape' && typeof node.params[f] === 'string') node.params[f] = node.params[f].replace(IDENT, t => (map.has(t) ? map.get(t) : t));
    if (node.type === 'block' && node.params.args) for (const k of Object.keys(node.params.args)) node.params.args[k] = String(node.params.args[k]).replace(IDENT, t => (map.has(t) ? map.get(t) : t));
  };
  // A use's inside, with sizes bound to the outer graph (for shape checking)
  function instantiateBlock(g, n, inS) {
    const blk = g.blocks[n.params.block];
    const map = bindBlock(g, n, inS);
    const inner = JSON.parse(JSON.stringify(blk.graph));
    inner.symbols = Object.assign({}, blk.graph.symbols, g.symbols); inner.blocks = g.blocks;
    const ins = inner.nodes.filter(z => z.type === 'input');
    ins.forEach((z, k) => { if (inS && inS[k]) z.params.shape = inS[k].join(', '); });
    for (const z of inner.nodes) substDims(z, map);
    return inner;
  }
  // Replace every block use with its layers, tagged as a group per use, so code generation writes one class per block.
  // Replace block uses by their layers. By default all the way down (for counting parameters and the variance plot);
  // { shallow: true } only the uses in g, leaving uses nested inside blocks as block nodes (for writing the code).
  function expandBlocks(g, o = {}) {
    const origin = new Map();
    if (!g.nodes.some(n => n.type === 'block')) return { flat: g, origin, outsOf: new Map() };
    const { shapes } = inferShapes(g), ix = index(g);
    const flat = { name: g.name, symbols: Object.assign({}, g.symbols), nodes: [], edges: [], groups: (g.groups || []).slice(), blocks: g.blocks };
    const outOf = new Map(), counts = {};
    for (const n of g.nodes) if (n.type !== 'block') flat.nodes.push(n);
    for (const n of g.nodes) {
      if (n.type !== 'block') continue;
      const blk = g.blocks && g.blocks[n.params.block]; if (!blk) continue;
      const outerSrc = ix.inKeys.get(n.id), inS = outerSrc.map(s => (s ? shapes.get(s) : null));
      const map = bindBlock(g, n, inS);
      const ins = blockInputs(blk), outNs = blockOutputs(blk);
      counts[blk.name] = (counts[blk.name] || 0) + 1;
      const hints = {}; for (const [k, v] of map) if (/^[A-Za-z_]\w*$/.test(v) && !(v in hints)) hints[v] = k;
      const base = blk.name.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase();
      // a repeated use becomes k chained copies named base_0 … base_{k-1}, which the generator writes as one nn.ModuleList
      // (copies chain through the first input; other inputs are shared by every copy)
      const k0 = Math.max(1, Math.floor(+n.params.repeat || 1)), reps = k0 > 1 && ins.length >= 1 && outNs.length === 1 ? k0 : 1;
      const attr = reps > 1 ? (n.params.name || base + 's' + (counts[blk.name] > 1 ? counts[blk.name] : ''))
        : (n.params.name || base + (/\d$/.test(base) ? '_' : '') + counts[blk.name]);
      let src = outerSrc.slice(), producers = [];
      for (let r = 0; r < reps; r++) {
        const pre = reps > 1 ? `${n.id}::${r}::` : `${n.id}::`;
        const idm = new Map(), members = [];
        for (const m of blk.graph.nodes) {
          if (m.type === 'input' || m.type === 'output') continue;
          const c = JSON.parse(JSON.stringify(m)); c.id = pre + m.id; substDims(c, map); if (c.params.tie) c.params.tie = pre + c.params.tie;
          flat.nodes.push(c); idm.set(m.id, c.id); members.push(c.id); origin.set(c.id, { use: n.id, inner: m.id, rep: r });
        }
        producers = [];
        for (const e of blk.graph.edges) {
          const k = ins.findIndex(z => z.id === e.from);
          const from = k >= 0 ? src[k] : e.out ? idm.get(e.from) + '@' + e.out : idm.get(e.from);
          const j = outNs.findIndex(z => z.id === e.to);
          if (j >= 0) { producers[j] = from; continue; }
          if (from && idm.has(e.to)) flat.edges.push({ id: pre + e.id, from, to: idm.get(e.to), port: e.port });
        }
        // the block's inputs and outputs in its own order, so its class keeps them (forward(self, x, h) → return y, h)
        flat.groups.push({ id: reps > 1 ? `${n.id}#${r}` : n.id, cls: blk.name, attr: reps > 1 ? `${attr}_${r}` : attr, label: attr, nodes: members, argHints: hints, argDefaults: Object.assign({}, blk.graph.symbols),
          blk, bind: Object.fromEntries(map), outs: outNs.map((_, j) => producers[j] || null), extIn: src.slice(), inNames: ins.map(z => z.params.name), outNames: outNs.map(z => z.params.name) });
        src = [producers[0], ...outerSrc.slice(1)];
      }
      outOf.set(n.id, producers);
    }
    for (const e of g.edges) { if (ix.byId.get(e.to) && ix.byId.get(e.to).type === 'block') continue; const c = Object.assign({}, e); c.from = srcKey(e); delete c.out; flat.edges.push(c); }
    // a key "use@k" becomes the layer inside the block that produces its output k (also through chained uses)
    // (a key stays "id@k" when it is output k of a block nested inside, which is still a block node)
    const resolve = f => { let i = 0; while (f && i++ < 1000) { const [id, k] = splitKey(f); if (!outOf.has(id)) return f; f = outOf.get(id)[k]; } return f || null; };
    for (const e of flat.edges) { const r = resolve(e.from); if (!r) { e.from = null; continue; } const [id, k] = splitKey(r); e.from = id; if (k) e.out = k; }
    flat.edges = flat.edges.filter(e => e.from);
    for (const gr of flat.groups) if (gr.outs) { gr.outs = gr.outs.map(resolve); gr.extIn = gr.extIn.map(f => (f ? resolve(f) : null)); }
    const outsOf = new Map([...outOf].map(([id, ps]) => [id, ps.map(p => (p ? resolve(p) : null))]));
    if (o.shallow || !flat.nodes.some(n => n.type === 'block')) return { flat, origin, outsOf };
    // deeper levels: a layer inside a nested block belongs to the outermost use
    const d = expandBlocks(flat), org = new Map(origin);
    for (const [id, o2] of d.origin) { const o1 = origin.get(o2.use); org.set(id, o1 ? { use: o1.use, inner: o2.inner, rep: o1.rep } : o2); }
    return { flat: d.flat, origin: org, outsOf };
  }
  // Class text for a block nobody uses yet (so every block still appears in the code)
  function blockClassCode(blk, opts, blocks) { const c = blockClass(blk, opts || {}, blocks); return c ? c.text : ''; }
  // { text, lines: inner node id → lines relative to the class line, syms: the block sizes its __init__ takes, in order }
  function blockClass(blk, opts, blocks, cache) {
    cache = cache || new Map();
    if (cache.has(blk.id)) return cache.get(blk.id);
    cache.set(blk.id, null); // a block that (somehow) contains itself gets no class rather than looping
    const G = { name: blk.name, symbols: Object.assign({ e: 8 }, blk.graph.symbols), nodes: blk.graph.nodes, edges: blk.graph.edges, groups: [], blocks: blocks || {} };
    if (!G.nodes.some(n => n.type === 'output') || !G.nodes.some(n => !['input', 'output'].includes(n.type))) return null;
    let r;
    try { r = generateModel(G, Object.assign({}, opts, { literalDims: false, className: blk.name, noMain: true, allBlocks: false, bcCache: cache })); } catch { return null; }
    const nm = pyIdent(blk.name, true), at = r.code.lastIndexOf('class ' + nm + '(');
    if (at < 0) return null;
    const first = r.code.slice(0, at).split('\n').length - 1, text = r.code.slice(at).replace(/\s+$/, ''), n = text.split('\n').length;
    const lines = new Map();
    for (const [k, ls] of r.lineMap) { const rel = ls.map(l => l - first).filter(l => l >= 0 && l < n); if (rel.length && !String(k).startsWith('g:')) lines.set(k, rel); }
    const deps = [...new Set(G.nodes.filter(z => z.type === 'block' && G.blocks[z.params.block]).map(z => z.params.block))];
    const res = { text, lines, syms: (r.mainSyms || []).slice(), deps };
    cache.set(blk.id, res);
    return res;
  }
  function generateModel(g, opts = {}) {
    // a block opened on its own is still a block: its sizes stay constructor arguments with defaults
    if (g.libBlock) opts = Object.assign({}, opts, { literalDims: false });
    const { flat, origin } = expandBlocks(g, { shallow: true });
    flat.blocks = g.blocks || {};
    const r = generateFlat(flat, Object.assign({ allBlocks: true }, opts));
    // map lines back to block uses and to the layers inside the block definition
    const lm = new Map(), put = (k, ls) => { if (!lm.has(k)) lm.set(k, []); lm.get(k).push(...ls); };
    for (const [k, ls] of r.lineMap) {
      put(k, ls);
      if (String(k).startsWith('g:')) put(String(k).slice(2).replace(/#\d+$/, ''), ls);
      const o = origin.get(k); if (o) { put(o.use, ls); put(o.inner, ls); }
    }
    for (const [k, ls] of lm) lm.set(k, [...new Set(ls)].sort((a, b) => a - b));
    r.lineMap = lm;
    return r;
  }
  // ── losses ──
  // Losses sit on the canvas but not in forward(): forward() returns the raw outputs, loss() scores them
  // against targets, and predict() adds the final activation the loss leaves out (softmax, sigmoid).
  // An input that only feeds losses is a target: it is an argument of loss(), not of forward().
  function lossParts(g) {
    const ix = index(g), losses = g.nodes.filter(n => n.type === 'loss');
    const targets = new Set(g.nodes.filter(n => n.type === 'input' && ix.outs.get(n.id).length && ix.outs.get(n.id).every(o => ix.byId.get(o.to) && ix.byId.get(o.to).type === 'loss')).map(n => n.id));
    const drop = new Set([...losses.map(n => n.id), ...targets]);
    const model = Object.assign({}, g, { nodes: g.nodes.filter(n => !drop.has(n.id)), edges: g.edges.filter(e => !drop.has(e.from) && !drop.has(e.to)) });
    return { ix, losses, targets, model };
  }
  // the parts of loss() and predict(): argument names and one expression per loss
  function lossPlan(g) {
    const { ix, losses, targets } = lossParts(g);
    if (!losses.length) return null;
    const { shapes } = inferShapes(g), sym = g.symbols || {};
    const outputs = g.nodes.filter(n => n.type === 'output');
    const fwdInputs = g.nodes.filter(n => n.type === 'input' && !targets.has(n.id));
    const outName = o => pyIdent(o.params.name || 'out');
    const args = outputs.map(outName), terms = [], warnings = [], lossArgs = [];
    const addArg = nm => { if (!lossArgs.includes(nm)) lossArgs.push(nm); return nm; };
    for (const L of losses) {
      const [ps, ts] = ix.ins.get(L.id), p = L.params, fn = LOSSES[p.fn] ? p.fn : 'cross_entropy', mode = fn === 'cross_entropy' ? p.mode || 'given' : 'given';
      const o = outputs.find(z => ix.ins.get(z.id)[0] === ps);
      if (!o) { warnings.push(`${LOSSES[fn].label}: connect it to a tensor that leaves the model (one that also goes to an Output).`); continue; }
      const tn = ts && ix.byId.get(ts);
      if (!tn) { warnings.push(`${LOSSES[fn].label}: it has no target.`); continue; }
      if (tn.type !== 'input') { warnings.push(`${LOSSES[fn].label}: its target must be an Input arrow.`); continue; }
      // a loss on probabilities would apply softmax/sigmoid twice
      const pn = ix.byId.get(ps);
      if (pn && pn.type === 'act' && ((fn === 'cross_entropy' && pn.params.fn === 'softmax') || (fn === 'bce' && pn.params.fn === 'sigmoid'))) warnings.push(`${LOSSES[fn].label} takes raw scores (logits), but this output already goes through ${pn.params.fn}. Remove that ${pn.params.fn}: predict() applies it at inference.`);
      const P = outName(o), Ps = shapes.get(ps) || [];
      const T = pyIdent(tn.params.name || 'target');
      let expr, doc;
      const kw = [];
      if (fn === 'cross_entropy') {
        if (+p.label_smoothing > 0) kw.push(`label_smoothing=${+p.label_smoothing}`);
        if (mode !== 'masked' && p.ignore_index !== '' && p.ignore_index != null && isFinite(+p.ignore_index)) kw.push(`ignore_index=${+p.ignore_index}`);
        const k = kw.length ? ', ' + kw.join(', ') : '';
        if (mode === 'next') { addArg(T); expr = `F.cross_entropy(${P}[:, :-1].flatten(0, -2), ${T}[:, 1:].flatten()${k})`; doc = `next-token cross-entropy: position t of ${P} is scored on token t + 1 of ${T}`; }
        else if (mode === 'masked') {
          addArg('labels');
          const pr = +p.mask_prob > 0 ? +p.mask_prob : 0.15, mid = p.mask_id !== '' && p.mask_id != null ? String(p.mask_id) : '';
          expr = `F.cross_entropy(${P}.flatten(0, -2), labels.flatten(), ignore_index=-100${k})`;
          doc = `masked-token cross-entropy: labels come from ${T}, masking ${+(pr * 100).toFixed(2)}% of them${mid ? ' with id ' + mid : ''} (-100 marks the rest)`;
        } else { addArg(T); expr = Ps.length > 2 ? `F.cross_entropy(${P}.flatten(0, -2), ${T}.flatten()${k})` : `F.cross_entropy(${P}, ${T}${k})`; doc = `cross-entropy of ${P} against the class indices in ${T}`; }
      } else {
        addArg(T);
        if (fn === 'bce') { expr = `F.binary_cross_entropy_with_logits(${P}, ${T}.float())`; doc = `binary cross-entropy of ${P} (logits) against ${T}`; }
        else if (fn === 'mse') { expr = `F.mse_loss(${P}, ${T})`; doc = `mean squared error between ${P} and ${T}`; }
        else if (fn === 'l1') { expr = `F.l1_loss(${P}, ${T})`; doc = `mean absolute error between ${P} and ${T}`; }
        else { expr = `F.huber_loss(${P}, ${T}, delta=${isFinite(+p.delta) && +p.delta > 0 ? +p.delta : 1.0})`; doc = `Huber loss between ${P} and ${T}`; }
      }
      const w = isFinite(+p.weight) && p.weight !== '' ? +p.weight : 1;
      terms.push({ id: L.id, expr: w === 1 ? expr : `${w} * ${expr}`, doc, fn, out: P, outputId: o.id, target: tn.id, targetName: mode === 'masked' ? 'labels' : T, mode, maskProb: +p.mask_prob > 0 ? +p.mask_prob : 0.15, maskId: p.mask_id });
    }
    return { args: args.concat(lossArgs.filter(a => !args.includes(a))), terms, warnings, outputs, fwdInputs, targets, ix, shapes, sym };
  }
  function generate(g, opts = {}) {
    const plan = lossPlan(g);
    if (!plan) return generateModel(g, opts);
    const r = generateModel(lossParts(g).model, opts);
    r.warnings = (r.warnings || []).concat(plan.warnings);
    if (!plan.terms.length) return r;
    const Q = '"' + '"' + '"';
    const lines = r.code.split('\n');
    const mainName = pyIdent(opts.className || g.name || 'Model', true);
    const c0 = lines.findIndex(l => l.startsWith('class ' + mainName + '(')); if (c0 < 0) return r;
    let end = c0 + 1; while (end < lines.length && (lines[end] === '' || /^\s/.test(lines[end]))) end++;
    while (end > c0 + 1 && lines[end - 1] === '') end--;
    // loss()
    const add = [], lmap = [];
    const cap = t => t.charAt(0).toUpperCase() + t.slice(1);
    add.push('', `    def loss(self, ${plan.args.join(', ')}):`);
    add.push(`        ${Q}${plan.terms.length === 1 ? cap(plan.terms[0].doc) + '.' : 'Sum of: ' + plan.terms.map(t => t.doc).join('; ') + '.'}${Q}`);
    if (plan.terms.length === 1) { lmap.push([plan.terms[0].id, add.length]); add.push(`        return ${plan.terms[0].expr}`); }
    else { add.push('        return ('); plan.terms.forEach((t, i) => { lmap.push([t.id, add.length]); add.push(`            ${i ? '+ ' : ''}${t.expr}`); }); add.push('        )'); }
    // predict(): forward() plus the activation each loss leaves out
    const fin = plan.fwdInputs.map(n => pyIdent(n.params.name || 'x'));
    const acts = plan.outputs.map(o => { const t = plan.terms.find(z => z.outputId === o.id); return t ? LOSSES[t.fn].act : null; });
    const wrap = (a, x) => (a === 'softmax' ? `torch.softmax(${x}, dim=-1)` : a === 'sigmoid' ? `torch.sigmoid(${x})` : x);
    const names = [...new Set(acts.filter(Boolean))];
    add.push('', '    @torch.no_grad()', `    def predict(self, ${fin.join(', ')}):`);
    add.push(`        ${Q}Inference: forward()${names.length ? ' followed by ' + names.join(' / ') + ', which the loss leaves out' : ' as is (the loss works on the raw output)'}.${Q}`);
    if (plan.outputs.length === 1) add.push(`        return ${wrap(acts[0], `self(${fin.join(', ')})`)}`);
    else { add.push(`        outs = self(${fin.join(', ')})`); add.push(`        return ${plan.outputs.map((o, i) => wrap(acts[i], `outs[${i}]`)).join(', ')}`); }
    lines.splice(end, 0, ...add);
    for (const [k, ls] of r.lineMap) r.lineMap.set(k, ls.map(l => (l >= end ? l + add.length : l)));
    for (const [id, i] of lmap) r.lineMap.set(id, [end + i]);
    // smoke test: also compute the loss
    const st = lines.findIndex(l => /^if __name__/.test(l));
    if (st >= 0) {
      const { ix, shapes, sym } = plan, dimsOf = sh => sh.map(d => (d === BATCH ? sym[BATCH] || 8 : evalDim(d, sym) ?? 8));
      const demo = n => {
        const dims = dimsOf(parseShape(n.params.shape));
        const emb = ix.outs.get(n.id).map(o => ix.byId.get(o.to)).find(c => c && c.type === 'embed' && !c.params.positions);
        if (emb) return `torch.randint(0, ${evalDim(emb.params.num, sym) ?? 100}, (${dims.join(', ')}${dims.length === 1 ? ',' : ''}))`;
        return `torch.randn(${dims.join(', ')})`;
      };
      const body = ['if __name__ == "__main__":', `    model = ${mainName}()`, '    print(sum(p.numel() for p in model.parameters()), "parameters")'];
      for (const n of plan.fwdInputs) body.push(`    ${pyIdent(n.params.name || 'x')} = ${demo(n)}`);
      const outN = plan.outputs.map(o => pyIdent(o.params.name || 'out'));
      body.push(`    ${outN.join(', ')} = model(${fin.join(', ')})`);
      for (const o of plan.outputs) { const sh = shapes.get(ix.ins.get(o.id)[0]); body.push(`    print(${pyIdent(o.params.name || 'out')}.shape)  # expected torch.Size([${(sh ? dimsOf(sh) : []).join(', ')}])`); }
      for (const a of plan.args.slice(plan.outputs.length)) {
        if (fin.includes(a)) continue;
        const t = plan.terms.find(z => z.targetName === a);
        const P = t && shapes.get(ix.ins.get(t.outputId)[0]), pd = P ? dimsOf(P) : [8];
        const tup = d => `(${d.join(', ')}${d.length === 1 ? ',' : ''})`;
        const ex = !t ? 'None' : t.fn === 'cross_entropy' ? `torch.randint(0, ${pd[pd.length - 1]}, ${tup(pd.slice(0, -1))})` : t.fn === 'bce' ? `torch.randint(0, 2, ${tup(pd)}).float()` : `torch.randn(${pd.join(', ')})`;
        body.push(`    ${a} = ${ex}`);
      }
      body.push(`    print(model.loss(${plan.args.join(', ')}).item(), "loss")`);
      lines.splice(st, lines.length - st, ...body, '');
    }
    r.code = lines.join('\n');
    return r;
  }
  // Turn imported submodule classes (groups) into block definitions with uses.
  // Module calls recorded while importing become blocks, innermost first: a module that calls other modules
  // becomes a block that contains their blocks.
  function groupsToBlocks(g) {
    let pending = (g.groups || []).slice();
    g.groups = [];
    const keep = { byKey: new Map(), names: new Set(Object.values(g.blocks || {}).map(b => b.name)) };
    while (pending.length) {
      const leaves = pending.filter(gr => !pending.some(o => o.parent === gr.id));
      pending = pending.filter(gr => !leaves.includes(gr));
      g.groups = leaves;
      const res = convertGroups(g, keep);
      g.groups = [];
      const remapKey = k => { let i = 0; while (k && res.remap.has(k) && i++ < 1000) { const [u, j] = res.remap.get(k); k = j ? u + '@' + j : u; } return k; };
      for (const gr of pending) {
        gr.nodes = gr.nodes.filter(id => !res.drop.has(id));
        for (const l of leaves) if (l.parent === gr.id) {
          if (res.useOf.has(l.id)) gr.nodes.push(res.useOf.get(l.id));
          else gr.nodes.push(...l.nodes.filter(id => !res.drop.has(id))); // a call that couldn't become a block stays inline
        }
        if (gr.outs) gr.outs = gr.outs.map(remapKey);
        if (gr.extIn) gr.extIn = gr.extIn.map(remapKey);
      }
    }
    collapseRepeats(g);
  }
  function convertGroups(g, keep) {
    const ix = index(g), ginfo = analyzeGroups(g, ix).filter(x => x.valid);
    const drop = new Set(), remap = new Map(), useOf = new Map();
    if (!ginfo.length) return { drop, remap, useOf };
    g.blocks = g.blocks || {};
    const { shapes } = inferShapes(g);
    const { byKey, names } = keep;
    const getDim = (n, f) => (f.startsWith('args.') ? (n.params.args || {})[f.slice(5)] : n.params[f]);
    for (const gi of ginfo) {
      const key = gi.group.cls + '|' + gi.ssig;
      const inDims = gi.extIn.map(s => { const sh = shapes.get(s); return sh ? sh[sh.length - 1] : null; });
      let ent = byKey.get(key);
      if (!ent) {
        let name = pyIdent(gi.group.cls, true), k = 2; while (names.has(name)) name = pyIdent(gi.group.cls, true) + k++;
        names.add(name);
        const nameOf = new Map(), taken = new Set([BATCH]);
        // a size without a hint of its own takes the name of the one constructor argument with the same value
        // (an input's width often arrives under the upstream name, while the class calls it c)
        const hints = gi.group.argHints || {}, byVal = t => { const v = evalDim(t, g.symbols); if (v == null) return null; const nms = [...new Set(Object.entries(hints).filter(([sym]) => evalDim(sym, g.symbols) === v).map(([, nm]) => nm))]; return nms.length === 1 ? nms[0] : null; };
        const shared = new Map();
        gi.ids.forEach(t => {
          const j = inDims.indexOf(t);
          const hint = hints[t] || (j >= 0 ? byVal(t) : null);
          if (hint && shared.has(hint)) { nameOf.set(t, shared.get(hint)); return; }
          let nm = hint || (j >= 0 ? (inDims.length > 1 ? 'd' + (j + 1) : 'd') : (t.replace(/[_\d]+$/, '') || 'h'));
          if (!hint && /^d\d*$/.test(nm) && j < 0) nm = 'h';
          while (taken.has(nm)) nm += '_';
          taken.add(nm); nameOf.set(t, nm); if (hint) shared.set(hint, nm);
        });
        // an input's width is the size the first layer reading it expects (the same number, maybe under another name)
        gi.extIn.forEach((s, j) => {
          const D = inDims[j]; if (!D || nameOf.has(D) || !/^[A-Za-z_]\w*$/.test(D)) return;
          for (const id of gi.order) {
            if (!ix.inKeys.get(id).includes(s)) continue;
            const n = ix.byId.get(id), f = n.type === 'linear' ? n.params.in : n.type === 'norm' ? n.params.dim : null;
            if (f && /^[A-Za-z_]\w*$/.test(f) && nameOf.has(f) && evalDim(f, g.symbols) != null && evalDim(f, g.symbols) === evalDim(D, g.symbols)) { nameOf.set(D, nameOf.get(f)); break; }
          }
        });
        const inner = { name, symbols: { e: g.symbols.e || 8 }, nodes: [], edges: [], groups: [] };
        const idm = new Map();
        const inNodes = gi.extIn.map((s, j) => {
          const sh = shapes.get(s) || ['e', 'd'];
          const dims = sh.map((d, i) => { if (d === BATCH) return BATCH; if (nameOf.has(d)) return nameOf.get(d); if (/^\d+$/.test(String(d))) return String(d); if (i === sh.length - 1) { const hv = byVal(d), nm = hv && ![...nameOf.values()].includes(hv) ? hv : inDims.length > 1 ? 'd' + (j + 1) : 'd'; nameOf.set(d, nm); return nm; }
            if (((sh.length === 3 && i === 1) || (sh.length === 2 && i === 0)) && d !== BATCH && ![...nameOf.values()].includes('t')) { nameOf.set(d, 't'); return 't'; } return d; });
          const z = makeNode('input', { name: gi.inNames ? gi.inNames[j] : gi.extIn.length > 1 ? 'x' + (j + 1) : 'x', shape: dims.join(', ') });
          inner.nodes.push(z); return z;
        });
        for (const [t, nm] of nameOf) inner.symbols[nm] = (gi.group.argDefaults && gi.group.argDefaults[nm] != null ? gi.group.argDefaults[nm] : evalDim(t, g.symbols)) ?? 64;
        for (const n of inner.nodes) parseShape(n.params.shape).forEach(d => identsIn(d).forEach(t => { if (inner.symbols[t] == null) inner.symbols[t] = g.symbols[t] ?? 16; }));
        for (const id of gi.order) {
          const c = JSON.parse(JSON.stringify(ix.byId.get(id))); c.id = uid('n');
          for (const f of DIMF[c.type] || []) if (typeof c.params[f] === 'string' && f !== 'shape') c.params[f] = c.params[f].replace(IDENT, t => nameOf.get(t) || t);
          if (c.type === 'block' && c.params.args) for (const k of Object.keys(c.params.args)) c.params.args[k] = String(c.params.args[k]).replace(IDENT, t => nameOf.get(t) || t);
          inner.nodes.push(c); idm.set(id, c.id);
        }
        // an edge inside the block, from a layer in it (or output k of a block in it) or from one of its inputs
        const innerEdge = (src, to, port) => {
          const [b, k] = splitKey(src);
          if (gi.members.has(b)) return Object.assign({ id: uid('e'), from: idm.get(b), to, port }, k ? { out: k } : {});
          return { id: uid('e'), from: inNodes[gi.extIn.indexOf(src)].id, to, port };
        };
        for (const id of gi.order) ix.inKeys.get(id).forEach((src, port) => { if (src) inner.edges.push(innerEdge(src, idm.get(id), port)); });
        for (const c of inner.nodes) if (c.params && c.params.tie) c.params.tie = idm.get(c.params.tie) || ''; // ties to layers outside the block are dropped
        gi.outputs.forEach((o, k) => {
          const nm = gi.outNames && gi.outNames[k] && /^[A-Za-z_]\w*$/.test(gi.outNames[k]) ? gi.outNames[k] : gi.outputs.length > 1 ? 'out' + (k + 1) : 'out';
          const on = makeNode('output', { name: nm }); inner.nodes.push(on);
          inner.edges.push(innerEdge(o, on.id, 0));
        });
        collapseRepeats(inner);
        autoLayout(inner);
        const blk = { id: uid('b'), name, graph: inner, labels: {} };
        g.blocks[blk.id] = blk;
        // where each size of the block appears, so another use can read its own value from the same place
        const dimsAt = gi.order.map(id => dimList(ix.byId.get(id)).map(([f, v]) => [f, String(v).replace(IDENT, t => nameOf.get(t) || t)]));
        ent = { blk, tmplIds: gi.ids, nameOf, dimsAt };
        byKey.set(key, ent);
      }
      // the use
      const args = {};
      for (const p of blockParams(ent.blk)) {
        ent.dimsAt.some((fs, i) => fs.some(([f, e]) => { if (e !== p) return false; args[p] = String(getDim(ix.byId.get(gi.order[i]), f)); return true; }));
      }
      let cx = 0, cy = 0; gi.members.forEach(id => { cx += ix.byId.get(id).x; cy += ix.byId.get(id).y; });
      const use = makeNode('block', { block: ent.blk.id, args, nIn: gi.extIn.length, nOut: gi.outputs.length, name: gi.group.attr || '' }, cx / gi.members.size, cy / gi.members.size);
      g.nodes.push(use); useOf.set(gi.group.id, use.id);
      gi.extIn.forEach((s, j) => { const [f, k] = splitKey(s); g.edges.push(Object.assign({ id: uid('e'), from: f, to: use.id, port: j }, k ? { out: k } : {})); });
      gi.outputs.forEach((o, k) => { if (gi.members.has(baseOf(o)) && !remap.has(o)) remap.set(o, [use.id, k]); });
      gi.members.forEach(id => drop.add(id));
    }
    // consumers of a replaced group's output now read from its block use (also when uses are chained)
    g.edges = g.edges.filter(e => !drop.has(e.to));
    for (const e of g.edges) { let i = 0, k = srcKey(e); while (remap.has(k) && i++ < 1000) { const [u, j] = remap.get(k); e.from = u; if (j) e.out = j; else delete e.out; k = srcKey(e); } }
    g.nodes = g.nodes.filter(n => !drop.has(n.id));
    g.edges = g.edges.filter(e => !drop.has(e.from));
    return { drop, remap, useOf };
  }
  function collapseRepeats(g) {
    const ix = index(g), byBase = new Map();
    for (const n of g.nodes) { if (n.type !== 'block') continue; const m = /^(.*)_(\d+)$/.exec(n.params.name || ''); if (!m) continue; const k = n.params.block + '|' + m[1]; if (!byBase.has(k)) byBase.set(k, []); byBase.get(k).push({ n, i: +m[2], base: m[1] }); }
    for (const arr of byBase.values()) {
      arr.sort((a, b) => a.i - b.i);
      if (arr.length < 2 || arr.some((x, j) => x.i !== j)) continue;
      const val = n => JSON.stringify(Object.entries(n.params.args || {}).sort().map(([p, v]) => [p, evalDim(v, g.symbols) ?? v]));
      // each reads the previous one's output first; any other inputs are the same for all
      const keys = x => ix.inKeys.get(x.n.id) || [];
      const chained = arr.every((x, j) => j === 0 || (keys(x)[0] === arr[j - 1].n.id && ix.outs.get(arr[j - 1].n.id).length === 1 && keys(x).length === keys(arr[0]).length && keys(x).slice(1).every((k, i) => k === keys(arr[0])[i + 1])));
      if (!chained || arr.some(x => val(x.n) !== val(arr[0].n))) continue;
      const first = arr[0].n, last = arr[arr.length - 1].n, rest = new Set(arr.slice(1).map(x => x.n.id));
      first.params.repeat = arr.length; first.params.name = arr[0].base;
      for (const e of g.edges) if (e.from === last.id) e.from = first.id;
      g.edges = g.edges.filter(e => !rest.has(e.to) && !rest.has(e.from));
      g.nodes = g.nodes.filter(n => !rest.has(n.id));
    }
  }

  const inputsOf = g => g.nodes.filter(n => n.type === 'input');
  function generateFlat(g, opts = {}) {
    opts = Object.assign({ fuseSilu: false, shapeComments: true, literalDims: true, className: g.name || 'Model' }, opts);
    const ix = index(g), sym = g.symbols || {};
    const { shapes, errors } = inferShapes(g);
    const ginfo = analyzeGroups(g, ix);
    const warnings = [];
    for (const gi of ginfo) if (!gi.valid) warnings.push(`Group “${gi.group.label || gi.group.attr}” ${gi.why}; its layers are written inline.`);
    for (const [id, m] of errors) warnings.push(`${nodeLabel(ix.byId.get(id))}: ${m}`);
    const valid = ginfo.filter(x => x.valid);
    const validOf = new Map(); for (const gi of valid) for (const id of gi.members) validOf.set(id, gi);

    // classes: every use of a block shares the block's own class; other groups by cls + signature
    const classes = []; // {name, sig, instances:[gi], args:[names]}
    const byBlock = new Map(), bcCache = opts.bcCache || new Map(), allBlocks = g.blocks || {};
    for (const gi of valid) {
      const blk = gi.group.blk; if (!blk) continue;
      let c = byBlock.get(blk.id);
      if (c === undefined) {
        const bc = blockClass(blk, opts, allBlocks, bcCache);
        c = bc ? { block: true, base: pyIdent(blk.name, true), name: pyIdent(blk.name, true), bc, instances: [] } : null;
        byBlock.set(blk.id, c); if (c) classes.push(c);
      }
      if (!c) continue;
      c.instances.push(gi); gi.cls = c;
      gi.ids = c.bc.syms.map(t => { const v = gi.group.bind && gi.group.bind[t]; return v != null ? String(v) : String(blk.graph.symbols[t] ?? t); });
    }
    for (const gi of valid) {
      if (gi.cls) continue;
      const base = pyIdent(gi.group.cls || 'Block', true);
      let c = classes.find(c => c.base === base && c.sig === gi.sig && c.ids.length === gi.ids.length);
      if (!c) {
        const sameBase = classes.filter(c => c.base === base).length;
        c = { base, name: sameBase ? base + (sameBase + 1) : base, sig: gi.sig, ids: gi.ids, instances: [] };
        classes.push(c);
      }
      c.instances.push(gi); gi.cls = c;
    }
    for (const c of classes) {
      if (c.block) continue;
      const used = new Set();
      c.args = c.ids.map((id0, k) => {
        const vals = c.instances.map(gi => gi.ids[k]);
        // class arguments get a generic name: h1/h2/h3 → h, f1 → f
        const hint = c.instances.map(gi => gi.group.argHints && gi.group.argHints[gi.ids[k]]).find(Boolean);
        let name = hint || id0.replace(/[_\d]+$/, '') || 'd';
        if (/^\d/.test(name) || PY_KW.has(name)) name = 'd' + k;
        while (used.has(name)) name = name + '_';
        used.add(name); return name;
      });
    }
    const subst = (expr, map) => {
      const r = String(expr).replace(IDENT, t => (map && map.has(t) ? map.get(t) : t));
      if (map && map.literal) { const v = evalDim(r, {}); return v != null ? String(v) : r; }
      return r;
    };

    const shapeStr = (nodeId, map) => {
      const s = shapes.get(nodeId); return s ? '(' + s.map(d => subst(d, map)).join(', ') + ')' : '';
    };

    // Emit one nn.Module given a scope.
    function emitModule(name, initSig, items, inputNames, outputRefs, map) {
      // items: [{id, kind:'node'|'call', node?, gi?, ins:[ref]}], refs: item id or 'in:k'
      const itemById = new Map(items.map(it => [it.id, it]));
      // a call with several outputs is referred to as "g:…@k" (its output k)
      const baseRef = r => (itemById.has(r) ? r : splitKey(String(r))[0]);
      const attrs = new Map(); const usedAttr = new Set(['training']);
      const defCount = {};
      for (const it of items) if (it.kind === 'node' && DEF_ATTR[it.node.type] && !it.node.params.name && (it.node.type !== 'func' || it.node.params.init)) defCount[it.node.type] = (defCount[it.node.type] || 0) + 1;
      const defSeen = {};
      const attrFor = it => {
        if (attrs.has(it.id)) return attrs.get(it.id);
        let a;
        if (it.kind === 'call') a = pyIdent(it.gi.group.attr || 'block');
        else if (it.kind === 'list') a = pyIdent(it.base);
        else if (it.node.params.name) a = pyIdent(it.node.params.name);
        else { const t = it.node.type, b = DEF_ATTR[t]; defSeen[t] = (defSeen[t] || 0) + 1; a = defCount[t] > 1 ? b + defSeen[t] : b; }
        let u = a, k = 2; while (usedAttr.has(u)) u = a + '_' + k++;
        usedAttr.add(u); attrs.set(it.id, u); return u;
      };
      const isModCall = it => it.kind === 'call' || it.kind === 'list' || ['linear', 'norm', 'dropout', 'embed'].includes(it.node.type) || (it.node.type === 'func' && it.node.params.init);

      // SiLU fusion
      const fused = new Set();
      if (opts.fuseSilu) for (const it of items) {
        if (it.kind !== 'node' || it.node.type !== 'binop' || it.node.params.op !== '*' || it.ins.length !== 2) continue;
        for (let k = 0; k < 2; k++) {
          const sg = itemById.get(it.ins[k]), other = it.ins[1 - k];
          if (sg && sg.kind === 'node' && sg.node.type === 'act' && sg.node.params.fn === 'sigmoid' && sg.ins[0] === other) {
            const cons = items.filter(o => o.ins.includes(sg.id)).length + outputRefs.filter(r => r === sg.id).length;
            if (cons === 1) { it.silu = other; it.ins = [other]; fused.add(sg.id); break; }
          }
        }
      }
      // reachable DFS post-order
      const order = [], seen = new Set();
      const dfs = r => {
        r = baseRef(r);
        if (seen.has(r) || String(r).startsWith('in:') || !itemById.has(r)) return; seen.add(r);
        for (const s of itemById.get(r).ins) dfs(s);
        order.push(r);
      };
      outputRefs.forEach(dfs);
      const dead = items.filter(it => !seen.has(it.id) && !fused.has(it.id));
      if (dead.length) warnings.push(`${dead.length} layer(s) in ${name} don't reach an output and were left out.`);
      const uses = new Map(), consumer = new Map();
      const addUse = (r, by) => { uses.set(r, (uses.get(r) || 0) + 1); consumer.set(r, by); };
      for (const id of order) for (const s of itemById.get(id).ins) addUse(s, id);
      for (const id of order) { const it = itemById.get(id); if (it.kind === 'node' && (it.node.type === 'scale' || (it.node.type === 'embed' && it.node.params.positions)) && it.ins[0]) addUse(it.ins[0], id); }
      outputRefs.forEach(r => addUse(r, '__out'));

      // init lines
      const init = [], initAt = new Map(), fwdAt = new Map();
      // sizes used inside forward() (e.g. the head count) are kept on self in named style: self.n_head = n_head
      const selfSyms = [];
      const fwdSize = expr => subst(expr, map).replace(IDENT, t => { if (!selfSyms.includes(t)) selfSyms.push(t); return 'self.' + t; });
      for (const it of items) if (it.kind === 'node' && it.node.type === 'heads') fwdSize(it.node.params.heads);
      for (const t of selfSyms) init.push(`self.${t} = ${t}`);
      const mark = (ids, ln) => ids.forEach(i => { if (!fwdAt.has(i)) fwdAt.set(i, []); fwdAt.get(i).push(ln); });
      for (const id of order) {
        const it = itemById.get(id);
        const start = init.length;
        if (it.kind === 'list') {
          const calls = it.gis.map(gi => `${gi.cls.name}(${gi.ids.map(t => subst(t, map)).join(', ')})`);
          const a = attrFor(it);
          if (calls.every(c => c === calls[0])) init.push(`self.${a} = nn.ModuleList([${calls[0]} for _ in range(${calls.length})])`);
          else if (calls.join(', ').length < 70) init.push(`self.${a} = nn.ModuleList([${calls.join(', ')}])`);
          else init.push(`self.${a} = nn.ModuleList([`, ...calls.map(c => `    ${c},`), `])`);
          initAt.set(id, [start, init.length]);
          continue;
        }
        if (it.kind === 'call') {
          const c = it.gi.cls; init.push(`self.${attrFor(it)} = ${c.name}(${it.gi.ids.map(t => subst(t, map)).join(', ')})`); continue;
        }
        const n = it.node, p = n.params, S = e => subst(e, map);
        if (n.type === 'linear') {
          const a = attrFor(it);
          init.push(`self.${a} = nn.Linear(${S(p.in)}, ${S(p.out)}${p.bias ? '' : ', bias=False'})`);
          const ic = p.tie ? null : initCode(p, `self.${a}.weight`);
          if (ic) init.push(ic);
          // a tied layer's weight comes from elsewhere, but its own bias is still zeroed when asked
          if (p.bias && (p.tie ? p.zero_bias === true : ic && p.zero_bias !== false)) init.push(`nn.init.zeros_(self.${a}.bias)`);
        }
        else if (n.type === 'norm') init.push(`self.${attrFor(it)} = ${normCtor(p, S(p.dim))}`);
        else if (n.type === 'dropout') init.push(`self.${attrFor(it)} = nn.Dropout(${p.p})`);
        else if (n.type === 'embed') { const a = attrFor(it); init.push(`self.${a} = nn.Embedding(${S(p.num)}, ${S(p.dim)})`); const ic = p.tie ? null : initCode(p, `self.${a}.weight`); if (ic) init.push(ic); }
        else if (n.type === 'func' && p.init) init.push(`self.${attrFor(it)} = ${S(p.init)}`);
        if (init.length > start) initAt.set(id, [start, init.length]);
      }
      // tied weights: one Parameter shared by two layers (set after both exist)
      for (const id of order) {
        const it = itemById.get(id); if (!it || it.kind !== 'node' || !it.node.params.tie) continue;
        const o = itemById.get(it.node.params.tie);
        if (!o || o.kind !== 'node' || !weightShape(o.node)) { warnings.push(`${attrFor(it)}: its tied layer isn't in the same module, so the tie was left out.`); continue; }
        const start = init.length; init.push(`self.${attrFor(it)}.weight = self.${attrFor(o)}.weight`);
        initAt.set(id + ':tie', [start, init.length]);
      }

      // forward
      const lines = [], expr = new Map();
      const live = new Map(); // var -> remaining uses
      const varOf = new Map();
      inputNames.forEach((nm, k) => { live.set(nm, uses.get('in:' + k) || 0); varOf.set('in:' + k, nm); });
      const ref = r => {
        if (varOf.has(r)) return { code: varOf.get(r), prec: 100, refs: [varOf.get(r)], ids: [] };
        return expr.get(r);
      };
      const wrap = (e, prec, right) => (e.prec < prec || (right && e.prec === prec) ? `(${e.code})` : e.code);
      const alloc = base => {
        base = pyIdent(base);
        if (!live.get(base)) return base;
        for (let k = 2; ; k++) if (!live.get(base + k)) return base + k;
      };
      const release = refs => { for (const v of refs) if (live.has(v)) live.set(v, live.get(v) - 1); };
      for (const id of order) {
        const it = itemById.get(id); const a = it.ins.map(ref);
        if (a.some(x => !x)) continue;
        if (it.kind === 'call' && it.gi.outputs.length > 1) {
          // a, b = self.cell(x, h): one variable per output that is used (unused ones are _)
          const ids = [id, ...a.flatMap(x => x.ids || [])];
          release(a.flatMap(x => x.refs));
          const names = it.gi.outNames || [];
          const vs = it.gi.outputs.map((_, k) => {
            const u = uses.get(id + '@' + k) || 0; if (!u) return '_';
            const nm = names[k] && /^[A-Za-z_]\w*$/.test(names[k]) && names[k] !== 'out' ? names[k] : 'y' + (k + 1);
            const v = alloc(nm); live.set(v, u); varOf.set(id + '@' + k, v); return v;
          });
          mark(ids, lines.length);
          const cm = opts.shapeComments ? `  # ${it.gi.outputs.map(o => shapeStr(o, map)).join(', ')}` : '';
          lines.push(`${vs.join(', ')} = self.${attrFor(it)}(${a.map(x => x.code).join(', ')})` + cm);
          continue;
        }
        if (it.kind === 'list') {
          const ids = [id, ...a.flatMap(x => x.ids || [])];
          // the shared inputs stay live through the loop: the loop variable can't take one of their names
          release(a[0].refs);
          const v = alloc('x');
          release(a.slice(1).flatMap(x => x.refs));
          if (a[0].code !== v) { mark(ids, lines.length); lines.push(`${v} = ${a[0].code}`); }
          mark(ids, lines.length);
          lines.push(`for blk in self.${attrFor(it)}:` + (opts.shapeComments ? `  # ${shapeStr(it.gis[it.gis.length - 1].output, map)}` : ''));
          mark(ids, lines.length); lines.push(`    ${v} = blk(${[v, ...a.slice(1).map(x => x.code)].join(', ')})`);
          live.set(v, uses.get(id) || 0); varOf.set(id, v);
          continue;
        }
        let code, prec = 100, refs = a.flatMap(x => x.refs);
        // the input appears twice in these expressions, so its variable is used twice
        if (it.kind === 'node' && (it.node.type === 'scale' || (it.node.type === 'embed' && it.node.params.positions)) && a[0]) refs = refs.concat(a[0].refs);
        if (it.kind === 'call') code = `self.${attrFor(it)}(${a.map(x => x.code).join(', ')})`;
        else {
          const n = it.node, p = n.params;
          if (it.silu) code = `F.silu(${a[0].code})`;
          else switch (n.type) {
            case 'norm': {
              const rank = (shapes.get(ix.ins.get(n.id)[0]) || []).length;
              code = NORMS[p.kind] && NORMS[p.kind].channelFirst && rank === 3
                ? `self.${attrFor(it)}(${wrap(a[0], 100)}.transpose(1, 2)).transpose(1, 2)`
                : `self.${attrFor(it)}(${a[0].code})`;
              break;
            }
            case 'linear': case 'dropout': code = `self.${attrFor(it)}(${a[0].code})`; break;
            case 'embed': code = p.positions ? `self.${attrFor(it)}(torch.arange(${wrap(a[0], 100)}.size(1), device=${wrap(a[0], 100)}.device))` : `self.${attrFor(it)}(${a[0].code})`; break;
            case 'heads': code = `split_heads(${a[0].code}, ${fwdSize(p.heads)})`; break;
            case 'merge': code = `merge_heads(${a[0].code})`; break;
            case 'transpose': code = `${wrap(a[0], 100)}.transpose(-2, -1)`; break;
            case 'scale': prec = 20; code = `${wrap(a[0], 20)} * ${wrap(a[0], 100)}.size(-1) ** -0.5`; break;
            case 'mask': code = p.mode === 'given' ? `apply_mask(${a[0].code}, ${a[1].code})` : `causal_mask(${a[0].code})`; break;
            case 'rope': {
              const base = +p.base === 10000 ? '' : `, ${p.base}`, secs = String(p.sections || '16, 24, 24').split(',').map(v => v.trim()).filter(Boolean);
              if (p.mode === 'axial') code = `apply_rope_axial(${a[0].code}, ${a[1].code}${base})`;
              else if (p.mode === 'mrope') code = `apply_rope_mrope(${a[0].code}, ${a[1].code}, ${p.base}${secs.join(', ') === '16, 24, 24' ? '' : `, (${secs.join(', ')})`})`;
              else code = `apply_rope(${a[0].code}${base})`;
              break;
            }
            case 'act': code = p.fn === 'softmax' ? `torch.softmax(${a[0].code}, dim=${p.dim ?? -1})` : (ACT_CODE[p.fn] || ACT_CODE.silu)(a[0].code); break;
            case 'binop': {
              const op = p.op || '+'; prec = OP_PREC[op] || 10;
              if (a.length === 1) code = p.rev ? `${p.const} ${op} ${wrap(a[0], prec, true)}` : `${wrap(a[0], prec)} ${op} ${p.const}`;
              else code = `${wrap(a[0], prec)} ${op} ${wrap(a[1], prec, true)}`;
              break;
            }
            case 'func':
              if (p.init) code = `self.${attrFor(it)}(${a.map(x => x.code).join(', ')})`;
              else if (String(p.fn).startsWith('.')) code = `${wrap(a[0], 100)}${p.fn}(${p.extra || ''})`;
              else if (/cat|stack/.test(p.fn)) code = `${p.fn}([${a.map(x => x.code).join(', ')}]${p.extra ? ', ' + p.extra : ''})`;
              else code = `${p.fn}(${a.map(x => x.code).join(', ')}${p.extra ? ', ' + p.extra : ''})`;
              break;
          }
        }
        const e = { code, prec, refs, ids: [id, ...a.flatMap(x => x.ids || [])], hasCall: isModCall(it) || a.some(x => x.hasCall), top: it.kind === 'node' && it.node.type === 'binop' ? 'op' : 'fn' };
        const u = uses.get(id) || 0, c = consumer.get(id), cit = itemById.get(c);
        // residual adds stay on their own line; a single-use value can be folded into its consumer (always into return)
        const isAdd = it.kind === 'node' && it.node.type === 'binop' && it.node.params.op === '+' && it.ins.length === 2;
        // a layer's result that feeds another layer gets its own line (x = F.relu(self.fc1(x)), not nested calls)
        const nests = cit && isModCall(cit) && e.hasCall && e.top !== 'op';
        const named = it.kind === 'node' && it.node.params.var && c !== '__out'; // a value named in the diagram gets its own line
        const inline = !named && u === 1 && !(cit && cit.kind === 'list') && (c === '__out' || (code.length <= 60 && !nests && !(isModCall(it) && cit && isModCall(cit)) && !(isAdd && cit && isModCall(cit))));
        if (inline) { expr.set(id, e); continue; }
        release(refs);
        const v = alloc(it.kind === 'node' ? DEF_VAR(it.node) : 'x');
        live.set(v, u); varOf.set(id, v);
        const sid = it.kind === 'call' ? it.gi.output : it.node.id;
        mark(e.ids, lines.length);
        lines.push(`${v} = ${code}` + (opts.shapeComments ? `  # ${shapeStr(sid, map)}` : ''));
      }
      const rets = outputRefs.map(ref).filter(Boolean);
      mark(rets.flatMap(r => r.ids || []), lines.length);
      lines.push('return ' + (rets.length ? rets.map(r => r.code).join(', ') : 'None'));
      const ind = s => '        ' + s;
      const map2 = new Map();
      const add = (id, ln) => { if (!map2.has(id)) map2.set(id, []); map2.get(id).push(ln); };
      for (const [id, [a, b]] of initAt) for (let k = a; k < b; k++) add(id, 3 + k);
      for (const [id, ls] of fwdAt) ls.forEach(l => add(id, 3 + init.length + 2 + l));
      return { map: map2, text: [
        `class ${name}(nn.Module):`,
        `    def __init__(${['self', ...initSig].join(', ')}):`,
        `        super().__init__()`,
        ...(init.length ? init.map(ind) : []),
        ``,
        `    def forward(${['self', ...inputNames].join(', ')}):`,
        ...lines.map(ind),
      ].join('\n') };
    }

    const out = ['import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '', ''];
    const allNodes = g.nodes.concat(...Object.values(g.blocks || {}).map(b => b.graph.nodes));
    const usedHelpers = opts.noMain ? [] : helpersFor(allNodes);
    if (usedHelpers.length) out.push(usedHelpers.join('\n\n\n'), '', '');
    const lineMap = new Map();
    const offset = () => (out.length ? out.join('\n').split('\n').length : 0);
    const putLines = (id, ls) => { if (!lineMap.has(id)) lineMap.set(id, []); lineMap.get(id).push(...ls); };
    // group classes
    // block classes: each written once, the blocks it uses before it, in the order the model uses them
    const topoAt = new Map(topo(g, ix).order.map((id, i) => [id, i]));
    const firstUse = c => Math.min(...c.instances.flatMap(gi => [...gi.members].map(id => topoAt.get(id) ?? 1e9)));
    classes.sort((a, b) => firstUse(a) - firstUse(b));
    const emittedBlocks = new Set();
    const emitBlock = id => {
      const blk = allBlocks[id]; if (!blk || emittedBlocks.has(id)) return; emittedBlocks.add(id);
      const bc = blockClass(blk, opts, allBlocks, bcCache); if (!bc) return;
      bc.deps.forEach(emitBlock);
      const off = offset();
      for (const [k, ls] of bc.lines) putLines(k, ls.map(l => l + off));
      out.push(bc.text, '', '');
    };
    for (const c of classes) {
      if (c.block) { emitBlock(c.instances[0].group.blk.id); continue; }
      const gi = c.instances[0];
      const map = new Map(gi.ids.map((t, k) => [t, c.args[k]]));
      let extNames = gi.extIn.length === 1 ? ['x'] : gi.extIn.map((_, k) => 'x' + (k + 1));
      // the block's own input names (forward(self, x, h)) when they are usable
      if (gi.inNames && gi.inNames.every(nm => /^[A-Za-z_]\w*$/.test(nm) && !PY_KW.has(nm) && nm !== 'self') && new Set(gi.inNames).size === gi.inNames.length) extNames = gi.inNames.map(nm => pyIdent(nm));
      const items = gi.order.map(id => ({ id, kind: 'node', node: ix.byId.get(id), ins: ix.inKeys.get(id).map(s => (s && gi.members.has(baseOf(s)) ? baseOf(s) : 'in:' + gi.extIn.indexOf(s))) }));
      // block sizes are defaults: def __init__(self, d=256, h=512)
      const defs = gi.group.argDefaults || null;
      const sigArgs = c.args.map(nm => (defs && defs[nm] != null && defs[nm] !== '' && !isNaN(+defs[nm]) ? `${nm}=${+defs[nm]}` : nm));
      const allDef = sigArgs.every(a => a.includes('=')), sig2 = allDef ? sigArgs : c.args;
      const off = offset(), m = emitModule(c.name, sig2, items, extNames, gi.outputs.map(o => (gi.members.has(o) ? o : 'in:' + gi.extIn.indexOf(o))), map);
      for (const inst of c.instances) inst.order.forEach((nid, i) => { const ls = m.map.get(gi.order[i]); if (ls) putLines(nid, ls.map(l => l + off)); });
      out.push(m.text, '', '');
    }
    // blocks the model doesn't use (yet) still get their class
    if (opts.allBlocks) Object.keys(allBlocks).forEach(emitBlock);
    // main scope
    const inputs = g.nodes.filter(n => n.type === 'input');
    const outputs = g.nodes.filter(n => n.type === 'output');
    const refOf = s => {
      if (!s) return null;
      const n = ix.byId.get(baseOf(s)); if (!n) return null;
      if (n.type === 'input') return 'in:' + inputs.indexOf(n);
      const gi = validOf.get(n.id); if (gi) return gi.outputs.length > 1 ? 'g:' + gi.group.id + '@' + gi.outputs.indexOf(s) : 'g:' + gi.group.id;
      return s;
    };
    let items = [];
    for (const n of g.nodes) {
      if (n.type === 'input' || n.type === 'output' || validOf.has(n.id)) continue;
      items.push({ id: n.id, kind: 'node', node: n, ins: ix.inKeys.get(n.id).map(refOf) });
    }
    // a block use passes one argument per block input, in the block's order
    for (const gi of valid) items.push({ id: 'g:' + gi.group.id, kind: 'call', gi, ins: (gi.group.blk && gi.cls && gi.cls.block ? gi.group.extIn : gi.extIn).map(refOf) });
    let outRefs = outputs.map(o => refOf(ix.inKeys.get(o.id)[0])).filter(Boolean);
    // Chains blocks_0 → blocks_1 → … become one nn.ModuleList, so state_dict keys stay blocks.0.*, blocks.1.*
    {
      const byBase = {};
      for (const it of items) if (it.kind === 'call') { const m = /^(.*?)_(\d+)$/.exec(it.gi.group.attr || ''); if (m) (byBase[m[1]] = byBase[m[1]] || []).push({ it, k: +m[2] }); }
      const taken = new Set(items.map(it => (it.kind === 'call' ? it.gi.group.attr : it.node && it.node.params.name)).filter(Boolean));
      const refCount = r => items.reduce((t, it) => t + it.ins.filter(x => x === r).length, 0) + outRefs.filter(x => x === r).length;
      for (const [base, arr] of Object.entries(byBase)) {
        arr.sort((a, b) => a.k - b.k);
        if (arr.length < 2 || arr.some((x, i) => x.k !== i) || taken.has(base) || !base) continue;
        const chain = arr.map(x => x.it);
        // each copy reads the previous one's output first; any other inputs are the same for all (shared)
        const ok = chain.every((c, i) => c.gi.outputs.length === 1 && c.ins.length >= 1 && (i === 0 || (c.ins[0] === chain[i - 1].id && refCount(chain[i - 1].id) === 1 && c.ins.length === chain[0].ins.length && c.ins.slice(1).every((r, k) => r === chain[0].ins[k + 1]))));
        if (!ok) continue;
        const L = { id: 'L:' + base, kind: 'list', base, gis: chain.map(c => c.gi), ins: chain[0].ins };
        const last = chain[chain.length - 1].id, drop = new Set(chain.map(c => c.id));
        items = items.filter(it => !drop.has(it.id)); items.push(L);
        for (const it of items) it.ins = it.ins.map(r => (r === last ? L.id : r));
        outRefs = outRefs.map(r => (r === last ? L.id : r));
      }
    }
    // symbols used in main scope
    const usedSyms = [];
    const note = e => identsIn(e).forEach(t => { if (!usedSyms.includes(t) && t !== BATCH) usedSyms.push(t); });
    const ordered = topo(g, ix).order.map(id => ix.byId.get(id));
    for (const n of ordered) {
      if (validOf.has(n.id) && n.type === 'block') { for (const [, v] of dimList(n)) note(v); continue; }
      for (const [, v] of dimList(n)) note(v);
      if (n.type === 'func' && n.params.init) note(n.params.init);
    }
    // arguments in the order the model lists its sizes (stable across a round trip), unknown names last
    // a block's class lists the sizes that come with its inputs first: __init__(self, d=64, h=128)
    const inSyms = new Set(opts.noMain ? inputsOf(g).flatMap(n => parseShape(n.params.shape).flatMap(identsIn)) : []);
    const symOrder = [...Object.keys(sym).filter(t => inSyms.has(t)), ...Object.keys(sym).filter(t => !inSyms.has(t))];
    // a block's class takes its sizes in the order they first appear (input sizes first), which reads back the same;
    // a model lists them in the order of its sizes
    if (opts.noMain) usedSyms.sort((a, b) => (inSyms.has(b) ? 1 : 0) - (inSyms.has(a) ? 1 : 0));
    else usedSyms.sort((a, b) => { const i = symOrder.indexOf(a), j = symOrder.indexOf(b); return (i < 0 ? 1e9 : i) - (j < 0 ? 1e9 : j); });
    const sig = opts.literalDims ? [] : usedSyms.map(t => (sym[t] != null && sym[t] !== '' ? `${t}=${sym[t]}` : t));
    // literal mode: the top-level model writes actual sizes (nn.Linear(256, 512)); block classes keep size arguments
    const LIT = new Map(Object.entries(sym).filter(([k, v]) => k !== BATCH && v !== '' && v != null && !isNaN(+v)).map(([k, v]) => [k, String(+v)]));
    LIT.literal = true;
    const mainName = pyIdent(opts.className, true);
    const off = offset(), m = emitModule(mainName, sig, items, inputs.map(n => pyIdent(n.params.name || 'x')), outRefs, opts.literalDims ? LIT : null);
    for (const [id, ls] of m.map) {
      const abs = ls.map(l => l + off);
      if (String(id).startsWith('L:')) { putLines(id, abs); for (const it of items) if (it.id === id) it.gis.forEach(gi => { putLines('g:' + gi.group.id, abs); gi.members.forEach(nid => putLines(nid, abs)); }); }
      else if (String(id).startsWith('g:')) { const gi = valid.find(x => 'g:' + x.group.id === id); putLines(id, abs); gi.members.forEach(nid => putLines(nid, abs)); }
      else putLines(id, abs);
    }
    out.push(m.text);
    // smoke test
    const demo = inputs.map(n => {
      const dims = parseShape(n.params.shape).map(d => (d === BATCH ? (sym[BATCH] || 8) : evalDim(d, sym)));
      if (!dims.every(d => d != null)) return null;
      // an input that feeds a token embedding is token ids
      const emb = ix.outs.get(n.id).map(o => ix.byId.get(o.to)).find(c => c && c.type === 'embed' && !c.params.positions);
      if (emb) { const v = evalDim(emb.params.num, sym); return `torch.randint(0, ${v ?? 100}, (${dims.join(', ')}${dims.length === 1 ? ',' : ''}))`; }
      return `torch.randn(${dims.join(', ')})`;
    });
    // an input can bring its own example (an image grid, ids with image tokens): then every input gets a line of
    // its own, with its shape and sizes in a comment that reads back exactly
    const custom = inputs.some(n => n.params.demo);
    inputs.forEach((n, i) => { if (n.params.demo) demo[i] = n.params.demo; });
    if (inputs.length && demo.every(Boolean) && !opts.noMain) {
      const outShape = outputs.map(o => { const s = ix.inKeys.get(o.id)[0]; const sh = s && shapes.get(s); return sh ? 'torch.Size([' + sh.map(d => (d === BATCH ? (sym[BATCH] || 8) : evalDim(d, sym) ?? d)).join(', ') + '])' : '?'; });
      const names = inputs.map(n => pyIdent(n.params.name || 'x'));
      const hint = n => '(' + parseShape(n.params.shape).map(d => { const v = d === BATCH ? (sym[BATCH] || 8) : evalDim(d, sym); return /^[A-Za-z_]\w*$/.test(d) ? `${d}=${v ?? '?'}` : String(d); }).join(', ') + ')';
      const call = custom ? names.join(', ') : demo.join(', ');
      out.push('', '', 'if __name__ == "__main__":', `    model = ${mainName}()`,
        `    print(sum(p.numel() for p in model.parameters()), "parameters")`,
        ...(custom ? inputs.map((n, i) => `    ${names[i]} = ${demo[i]}  # ${hint(n)}`) : []),
        ...(outputs.length > 1
          ? [`    outs = model(${call})`, `    print([o.shape for o in outs])  # expected [${outShape.join(', ')}]`]
          : [`    out = model(${call})`, `    print(out.shape)  # expected ${outShape.join(', ')}`]));
    }
    return { code: out.join('\n') + '\n', warnings, lineMap, mainSyms: usedSyms.slice(), classes: classes.map(c => ({ name: c.name, ids: c.ids || [], args: c.args || [] })) };
  }
  function pyIdent(s, cls) {
    let t = String(s || '').trim().replace(/[^\w]/g, '_').replace(/^(\d)/, '_$1') || (cls ? 'Model' : 'x');
    if (cls) t = t.charAt(0).toUpperCase() + t.slice(1);
    if (PY_KW.has(t)) t += '_';
    return t;
  }
  function nodeLabel(n) {
    if (!n) return '?';
    const p = n.params;
    if (p.title) return p.title;
    if (n.type === 'block') return 'Block ' + (p.name || '');
    if (n.type === 'embed') return (p.positions ? 'Position embedding ' : 'Embedding ') + (p.name || '');
    if (n.type === 'loss') return LOSSES[p.fn] ? LOSSES[p.fn].label : 'Loss';
    if (['heads', 'merge', 'transpose', 'scale', 'mask', 'rope'].includes(n.type)) return { heads: 'Split heads', merge: 'Merge heads', transpose: 'Transpose', scale: 'Scale', mask: 'Causal mask', rope: 'Rotary embedding' }[n.type];
    return ({ input: 'Input ' + p.name, output: 'Output', linear: `Linear ${p.out}×${p.in}`, norm: p.kind, act: p.fn, dropout: 'Dropout', binop: p.op, func: p.fn || p.init })[n.type];
  }

  // ───────────────────────── Python importer ─────────────────────────
  function tokenize(src) {
    const toks = [], ind = [0]; let i = 0, line = 1, depth = 0, bol = true;
    const push = (t, v) => toks.push({ t, v, line });
    while (i < src.length) {
      if (bol && depth === 0) {
        let w = 0; while (src[i] === ' ' || src[i] === '\t') { w += src[i] === '\t' ? 4 : 1; i++; }
        if (src[i] === '\n' || src[i] === '#' || src[i] === '\r' || i >= src.length) { while (i < src.length && src[i] !== '\n') i++; i++; line++; continue; }
        if (w > ind[ind.length - 1]) { ind.push(w); push('indent'); }
        while (w < ind[ind.length - 1]) { ind.pop(); push('dedent'); }
        bol = false;
      }
      const c = src[i];
      if (c === '\n') { if (depth === 0) { push('nl'); bol = true; } i++; line++; continue; }
      if (c === ' ' || c === '\t' || c === '\r') { i++; continue; }
      if (c === '\\' && src[i + 1] === '\n') { i += 2; line++; continue; }
      if (c === '#') { while (i < src.length && src[i] !== '\n') i++; continue; }
      let m;
      const rest = src.slice(i, i + 400);
      if ((m = /^[rRbBfFuU]{0,2}("""|''')/.exec(rest)) || (m = /^[rRbBfFuU]{0,2}("""|''')/.exec(src.slice(i, i + 5)))) {
        const q = m[1]; const start = i + m[0].length; const end = src.indexOf(q, start);
        const s = src.slice(start, end < 0 ? src.length : end); line += (s.match(/\n/g) || []).length;
        push('str', s); i = end < 0 ? src.length : end + 3; continue;
      }
      if ((m = /^[rRbBfFuU]{0,2}(["'])/.exec(rest))) {
        const q = m[1]; let j = i + m[0].length, s = '';
        while (j < src.length && src[j] !== q && src[j] !== '\n') { if (src[j] === '\\') { s += src[j + 1]; j += 2; } else s += src[j++]; }
        push('str', s); i = j + 1; continue;
      }
      if ((m = /^(\d[\d_]*\.?\d*(e[-+]?\d+)?|\.\d+)/i.exec(rest))) { push('num', +m[0].replace(/_/g, '')); i += m[0].length; continue; }
      if ((m = /^[A-Za-z_]\w*/.exec(rest))) { push('name', m[0]); i += m[0].length; continue; }
      if ((m = /^(\*\*=?|\/\/=?|->|==|!=|<=|>=|[+\-*/@%&|^]=|<<|>>|\.\.\.)/.exec(rest))) { push('op', m[0]); i += m[0].length; continue; }
      if ('([{'.includes(c)) depth++;
      if (')]}'.includes(c)) depth = Math.max(0, depth - 1);
      push('op', c); i++;
    }
    push('nl'); while (ind.length > 1) { ind.pop(); push('dedent'); } push('eof');
    return toks;
  }

  function parse(src) {
    const T = tokenize(src); let k = 0;
    const peek = (o = 0) => T[k + o], next = () => T[k++];
    const is = (t, v) => peek().t === t && (v === undefined || peek().v === v);
    const isOp = v => is('op', v), isKw = v => is('name', v);
    const expect = (t, v) => { if (!is(t, v)) throw new SyntaxError(`Line ${peek().line}: expected ${v || t}, found ${peek().v ?? peek().t}`); return next(); };
    function block() {
      expect('op', ':');
      if (is('nl')) { next(); expect('indent'); const body = []; while (!is('dedent') && !is('eof')) { const s = stmt(); if (s) body.push(s); } if (is('dedent')) next(); return body; }
      const s = simple(); return s ? [s] : [];
    }
    function skipLine() { while (!is('nl') && !is('eof')) next(); if (is('nl')) next(); }
    function stmt() {
      if (is('nl')) { next(); return null; }
      if (isOp('@')) { skipLine(); return null; }
      if (isKw('class')) {
        next(); const name = expect('name').v; let bases = [];
        if (isOp('(')) { next(); while (!isOp(')')) { bases.push(expr()); if (isOp(',')) next(); } next(); }
        return { k: 'class', name, bases, body: block() };
      }
      if (isKw('def')) {
        next(); const name = expect('name').v; expect('op', '('); const params = [];
        while (!isOp(')')) {
          if (isOp('*') || isOp('**')) { next(); if (is('name')) next(); }
          else if (isOp('/')) next();
          else { const pn = expect('name').v; let def; if (isOp(':')) { next(); expr(); } if (isOp('=')) { next(); def = expr(); } params.push({ name: pn, def }); }
          if (isOp(',')) next();
        }
        next(); if (isOp('->')) { next(); expr(); }
        return { k: 'def', name, params, body: block() };
      }
      if (isKw('for')) { next(); const target = exprList(true); expect('name', 'in'); const iter = exprList(); return { k: 'for', target, iter, body: block() }; }
      if (isKw('if')) {
        next(); const test = expr(); const body = block(); let orelse = [];
        if (isKw('elif')) { T[k].v = 'if'; orelse = [stmt()]; }
        else if (isKw('else')) { next(); orelse = block(); }
        return { k: 'if', test, body, orelse };
      }
      if (isKw('with') || isKw('while') || isKw('try')) { next(); while (!isOp(':')) next(); const body = block(); while (isKw('except') || isKw('finally') || isKw('else')) { next(); while (!isOp(':')) next(); block(); } return { k: 'block', body }; }
      return simple();
    }
    function simple() {
      let s;
      if (isKw('import') || isKw('from') || isKw('pass') || isKw('assert') || isKw('raise') || isKw('global') || isKw('del')) { skipLine(); return null; }
      if (isKw('return')) { next(); s = { k: 'return', value: is('nl') ? null : exprList() }; }
      else {
        const e = exprList();
        if (isOp('=')) {
          const targets = [e]; let value;
          while (isOp('=')) { next(); value = exprList(); targets.push(value); }
          targets.pop(); s = { k: 'assign', targets, value };
        } else if (is('op') && /^[+\-*/@]=$/.test(peek().v)) { const op = next().v[0]; s = { k: 'assign', targets: [e], value: { k: 'bin', op, l: e, r: exprList() } }; }
        else if (isOp(':')) { next(); expr(); if (isOp('=')) { next(); s = { k: 'assign', targets: [e], value: exprList() }; } else s = null; }
        else s = { k: 'expr', value: e };
      }
      if (isOp(';')) next();
      if (is('nl')) next();
      return s;
    }
    function exprList(noIn) {
      const first = expr(noIn); if (!isOp(',')) return first;
      const items = [first]; while (isOp(',')) { next(); if (is('nl') || isOp('=') || isOp(')') || isKw('in')) break; items.push(expr(noIn)); }
      return { k: 'tuple', items };
    }
    const BIN = { or: 1, and: 2, '|': 4, '^': 5, '&': 6, '<<': 7, '>>': 7, '+': 8, '-': 8, '*': 9, '/': 9, '//': 9, '%': 9, '@': 9 };
    const CMP = new Set(['<', '>', '==', '!=', '<=', '>=', 'in', 'is']);
    function expr(noIn) {
      if (isKw('lambda')) { while (!isOp(':')) next(); next(); expr(); return { k: 'const', v: null }; }
      let e = binary(0, noIn);
      if (isKw('if')) { next(); const test = binary(0, noIn); expect('name', 'else'); const alt = expr(noIn); return { k: 'ifexp', test, body: e, alt }; }
      return e;
    }
    function binary(minP, noIn) {
      let l = unary(noIn);
      for (;;) {
        const t = peek(); let op = t.v;
        if (t.t === 'name' && op === 'not' && T[k + 1].v === 'in') { if (noIn) break; next(); next(); l = { k: 'cmp', op: 'not in', l, r: binary(3, noIn) }; continue; }
        if ((t.t === 'op' || t.t === 'name') && CMP.has(op) && minP <= 3) {
          if (noIn && op === 'in') break;
          next(); if (op === 'is' && isKw('not')) { next(); op = 'is not'; }
          l = { k: 'cmp', op, l, r: binary(4, noIn) }; continue;
        }
        const p = (t.t === 'op' || t.t === 'name') ? BIN[op] : undefined;
        if (p === undefined || p < minP || (t.t === 'name' && op !== 'and' && op !== 'or')) break;
        next(); l = { k: 'bin', op, l, r: binary(p + 1, noIn) };
      }
      return l;
    }
    function unary(noIn) {
      if (isOp('-') || isOp('+') || isOp('~')) { const op = next().v; return { k: 'unary', op, v: unary(noIn) }; }
      if (isKw('not')) { next(); return { k: 'unary', op: 'not', v: binary(3, noIn) }; }
      let e = postfix();
      if (isOp('**')) { next(); e = { k: 'bin', op: '**', l: e, r: unary(noIn) }; }
      return e;
    }
    function args() {
      const pos = [], kw = {};
      while (!isOp(')')) {
        if (isOp('*') || isOp('**')) { next(); pos.push({ k: 'star', v: expr() }); }
        else if (is('name') && T[k + 1].t === 'op' && T[k + 1].v === '=') { const n = next().v; next(); kw[n] = expr(); }
        else {
          const e = expr();
          if (isKw('for')) { pos.push(comp(e, ')')); break; }
          pos.push(e);
        }
        if (isOp(',')) next();
      }
      next(); return { pos, kw };
    }
    function comp(elt, close) {
      next(); const target = exprList(true); expect('name', 'in'); const iter = binary(0, true);
      let cond = null; if (isKw('if')) { next(); cond = binary(0); }
      if (close !== ')') expect('op', close);
      return { k: 'comp', elt, target, iter, cond };
    }
    function postfix() {
      let e = atom();
      for (;;) {
        if (isOp('.')) { next(); e = { k: 'attr', v: e, name: expect('name').v }; }
        else if (isOp('(')) { next(); const a = args(); e = { k: 'call', fn: e, args: a.pos, kw: a.kw }; }
        else if (isOp('[')) {
          next(); const idx = [];
          while (!isOp(']')) { if (isOp(':')) { next(); idx.push({ k: 'slice' }); } else { const x = expr(); if (isOp(':')) { next(); if (!isOp(']') && !isOp(',')) expr(); idx.push({ k: 'slice' }); } else idx.push(x); } if (isOp(',')) next(); }
          next(); e = { k: 'sub', v: e, idx };
        } else break;
      }
      return e;
    }
    function atom() {
      const t = next();
      if (t.t === 'num') return { k: 'const', v: t.v };
      if (t.t === 'str') { let s = t.v; while (is('str')) s += next().v; return { k: 'str', v: s }; }
      if (t.t === 'name') {
        if (t.v === 'True') return { k: 'const', v: true };
        if (t.v === 'False') return { k: 'const', v: false };
        if (t.v === 'None') return { k: 'const', v: null };
        return { k: 'name', v: t.v };
      }
      if (t.v === '...') return { k: 'const', v: null };
      if (t.v === '(') {
        if (isOp(')')) { next(); return { k: 'tuple', items: [] }; }
        const e = expr(); if (isKw('for')) { const c = comp(e, ')'); next(); return c; }
        if (isOp(',')) { const items = [e]; while (isOp(',')) { next(); if (isOp(')')) break; items.push(expr()); } next(); return { k: 'tuple', items }; }
        expect('op', ')'); return e;
      }
      if (t.v === '[') {
        if (isOp(']')) { next(); return { k: 'list', items: [] }; }
        const e = expr(); if (isKw('for')) return comp(e, ']');
        const items = [e]; while (isOp(',')) { next(); if (isOp(']')) break; items.push(expr()); }
        expect('op', ']'); return { k: 'list', items };
      }
      if (t.v === '{') { let d = 1; while (d && !is('eof')) { const x = next(); if (x.v === '{') d++; if (x.v === '}') d--; } return { k: 'const', v: null }; }
      throw new SyntaxError(`Line ${t.line}: unexpected ${t.v ?? t.t}`);
    }
    const body = []; while (!is('eof')) { if (is('dedent') || is('indent')) { next(); continue; } const s = stmt(); if (s) body.push(s); }
    return body;
  }

  function unparse(e, env) {
    if (!e) return '';
    const u = x => unparse(x, env);
    switch (e.k) {
      case 'const': return e.v === null ? 'None' : e.v === true ? 'True' : e.v === false ? 'False' : String(e.v);
      case 'str': return JSON.stringify(e.v);
      case 'name': { if (env && env.has(e.v)) { const v = env.get(e.v); if (typeof v === 'number') return String(v); if (v && v.sym) return v.sym; if (v && v.list) return '(' + v.list.map(valStr).join(', ') + (v.list.length === 1 ? ',)' : ')'); } return e.v; }
      case 'attr': return u(e.v) + '.' + e.name;
      case 'call': return u(e.fn) + '(' + [...e.args.map(u), ...Object.entries(e.kw).map(([a, b]) => a + '=' + u(b))].join(', ') + ')';
      case 'bin': return u(e.l) + ' ' + e.op + ' ' + u(e.r);
      case 'unary': return e.op + (e.op === 'not' ? ' ' : '') + u(e.v);
      case 'tuple': return '(' + e.items.map(u).join(', ') + (e.items.length === 1 ? ',)' : ')');
      case 'list': return '[' + e.items.map(u).join(', ') + ']';
      case 'sub': return u(e.v) + '[' + e.idx.map(i => (i.k === 'slice' ? ':' : u(i))).join(', ') + ']';
      case 'star': return '*' + u(e.v);
      default: return '…';
    }
  }
  const valStr = v => (typeof v === 'number' ? (v !== 0 && Math.abs(v) < 1e-3 ? v.toExponential().replace(/\.?0+e/, 'e').replace('e-0', 'e-') : String(v)) : v && v.sym ? v.sym : v === true ? 'True' : v === false ? 'False' : v == null ? 'None' : v.str != null ? JSON.stringify(v.str) : '?');
  const pathOf = e => (e.k === 'name' ? e.v : e.k === 'attr' ? (pathOf(e.v) ? pathOf(e.v) + '.' + e.name : null) : null);

  // a method's parameters, docstring and body lines, read straight from the source text
  function findMethod(src, cls, name) {
    const L = String(src).replace(/\r\n/g, '\n').split('\n');
    const c = L.findIndex(l => new RegExp('^class\\s+' + cls + '\\b').test(l)); if (c < 0) return null;
    let i = c + 1, hit = -1, ind = 0;
    for (; i < L.length; i++) {
      if (L[i].trim() && !/^\s/.test(L[i])) break;
      const m = new RegExp('^(\\s+)def\\s+' + name + '\\s*\\(\\s*self\\s*(?:,([^)]*))?\\)\\s*:').exec(L[i]);
      if (m) { hit = i; ind = m[1].length; break; }
    }
    if (hit < 0) return null;
    const params = (/\(\s*self\s*(?:,([^)]*))?\)/.exec(L[hit])[1] || '').split(',').map(s => s.trim().split(/[:=]/)[0].trim()).filter(Boolean);
    const body = [];
    for (let j = hit + 1; j < L.length; j++) { const l = L[j]; if (l.trim() && (l.length - l.trimStart().length) <= ind) break; if (l.trim()) body.push(l.trim()); }
    let doc = '';
    if (body.length && /^("""|''')/.test(body[0])) { const q = body[0].slice(0, 3); let k = 0, txt = body[0].slice(3); while (!txt.includes(q) && k + 1 < body.length) { k++; txt += ' ' + body[k]; } doc = txt.slice(0, txt.indexOf(q)); body.splice(0, k + 1); }
    return { params, doc, body: body.filter(l => !l.startsWith('#')) };
  }
  function importPython(src, opts = {}) {
    const ast = parse(src);
    const classes = new Map();
    for (const s of ast) if (s.k === 'class') classes.set(s.name, s);
    if (!classes.size) throw new Error('No class definitions found. Paste an nn.Module subclass.');
    // pick the root: a class with forward() not constructed by any other class
    const constructed = new Set();
    const scan = e => { if (!e || typeof e !== 'object') return; if (e.k === 'call') { const p = pathOf(e.fn); if (p && classes.has(p)) constructed.add(p); } for (const v of Object.values(e)) if (Array.isArray(v)) v.forEach(scan); else if (v && typeof v === 'object') scan(v); };
    for (const c of classes.values()) scan(c.body);
    const hasFwd = c => c.body.some(s => s.k === 'def' && s.name === 'forward');
    let roots = [...classes.values()].filter(c => hasFwd(c) && !constructed.has(c.name));
    if (opts.root && classes.has(opts.root)) roots = [classes.get(opts.root)];
    if (!roots.length) throw new Error('Could not find a top-level module with a forward() method.');
    const root = roots[roots.length - 1];
    const usedClasses = new Set([root.name]);

    const g = { name: root.name, symbols: { [BATCH]: 8 }, nodes: [], edges: [], groups: [] };
    const warnings = [], tiedSpecs = [];
    const symCount = {};
    const newSym = (base, value) => {
      base = base.replace(/\W/g, '_');
      symCount[base] = (symCount[base] || 0) + 1;
      let name = symCount[base] === 1 ? base : base + symCount[base];
      while (g.symbols[name] != null && g.symbols[name] !== value) name += '_';
      g.symbols[name] = value; return { sym: name };
    };
    let curGroup = null, seqInput = false;
    const addNode = (type, params, inputs) => {
      const n = makeNode(type, params);
      g.nodes.push(n);
      inputs.forEach((t, port) => g.edges.push({ id: uid('e'), from: t.t, to: n.id, port }));
      if (curGroup) curGroup.nodes.push(n.id);
      return { t: n.id };
    };
    const isT = v => v && v.t != null;
    const num = v => (typeof v === 'number' ? v : null);
    const dimStr = v => (typeof v === 'number' ? String(v) : v && v.sym ? v.sym : v && v.list ? valStr(v.list[v.list.length - 1]) : String(v));

    function arith(op, a, b) {
      if (op === '**' && a && a.sizeOf && b === -0.5) return { rsqrtOf: a.sizeOf };
      if (typeof a === 'number' && typeof b === 'number') {
        switch (op) { case '+': return a + b; case '-': return a - b; case '*': return a * b; case '/': return a / b; case '//': return Math.floor(a / b); case '%': return a % b; case '**': return a ** b; }
      }
      if (a && a.list && op === '+' && b && b.list) return { list: a.list.concat(b.list) };
      const s = v => { const t = valStr(v); return /[\s+\-]/.test(t) ? '(' + t + ')' : t; };
      if ((a && a.sym) || (b && b.sym)) return { sym: `${s(a)}${op === '//' ? '//' : op}${s(b)}`.replace(/\*\*/, '**') };
      return null;
    }

    function ev(e, env) {
      switch (e.k) {
        case 'const': return e.v;
        case 'str': return { str: e.v };
        case 'name': if (env.has(e.v)) return env.get(e.v); return { path: e.v };
        case 'attr': {
          const base = ev(e.v, env);
          if (base && base.inst) { if (e.name in base.inst.attrs) return base.inst.attrs[e.name]; return { path: 'self.' + e.name, missing: true }; }
          if (isT(base)) return { method: e.name, self: base };
          if (base && base.mod && (e.name === 'weight' || e.name === 'bias')) return { param: e.name, of: base };
          if (base && base.path) return { path: base.path + '.' + e.name };
          return { path: '?.' + e.name };
        }
        case 'tuple': case 'list': return { list: e.items.flatMap(x => (x.k === 'star' ? (ev(x.v, env).list || []) : [ev(x, env)])) };
        case 'comp': {
          const it = ev(e.iter, env); const seq = it && it.list ? it.list : [];
          return { list: seq.flatMap(v => { const env2 = new Map(env); bind(e.target, v, env2); if (e.cond && !truthy(ev(e.cond, env2))) return []; return [ev(e.elt, env2)]; }) };
        }
        case 'unary': {
          const v = ev(e.v, env);
          if (e.op === 'not') return !truthy(v);
          if (e.op === '-') { if (typeof v === 'number') return -v; if (v && v.sym) return { sym: '-' + v.sym }; if (isT(v)) return addNode('binop', { op: '*', const: '-1' }, [v]); }
          return v;
        }
        case 'bin': {
          const a = ev(e.l, env), b = ev(e.r, env);
          if (e.op === 'and') return truthy(a) ? b : a;
          if (e.op === 'or') return truthy(a) ? a : b;
          if (e.op === '*' && isT(a) && b && b.rsqrtOf && b.rsqrtOf.t === a.t) return addNode('scale', {}, [a]); // q * q.size(-1) ** -0.5
          if (isT(a) || isT(b)) {
            const op = ['+', '-', '*', '/', '@'].includes(e.op) ? e.op : null;
            if (!op) { warnings.push(`Operator ${e.op} on tensors is not supported; treated as +.`); }
            if (isT(a) && isT(b)) return addNode('binop', { op: op || '+' }, [a, b]);
            if (isT(a)) return addNode('binop', { op: op || '+', const: valStr(b) }, [a]);
            return addNode('binop', { op: op || '+', const: valStr(a), rev: true }, [b]);
          }
          return arith(e.op, a, b);
        }
        case 'cmp': {
          const a = num2(ev(e.l, env)), b = num2(ev(e.r, env));
          if (typeof a === 'number' && typeof b === 'number') return { '<': a < b, '>': a > b, '==': a === b, '!=': a !== b, '<=': a <= b, '>=': a >= b }[e.op] ?? null;
          if (e.op === 'is') return a === b || (a == null && b == null);
          if (e.op === 'is not') return !(a === b || (a == null && b == null));
          return null;
        }
        case 'ifexp': { const t = ev(e.test, env); return truthy(t) ? ev(e.body, env) : ev(e.alt, env); }
        case 'sub': {
          const v = ev(e.v, env), i = e.idx.length === 1 ? ev(e.idx[0], env) : null;
          if (v && v.list && typeof i === 'number') return v.list[i < 0 ? v.list.length + i : i];
          if (isT(v)) { warnings.push('Tensor indexing was kept as a custom op.'); return addNode('func', { fn: '.__getitem__', extra: e.idx.map(x => (x.k === 'slice' ? 'slice(None)' : unparse(x, env))).join(', '), n: 1 }, [v]); }
          return null;
        }
        case 'call': return call(e, env);
        default: return null;
      }
    }
    // numeric value of a symbol bound to a known size (e.g. n=3 in __init__)
    const num2 = v => (typeof v === 'number' ? v : v && v.sym ? evalDim(v.sym, g.symbols) ?? v : v);
    const truthy = v => !(v === false || v == null || v === 0 || (v && v.list && !v.list.length));
    function bind(target, v, env) {
      if (target.k === 'name') env.set(target.v, v);
      else if (target.k === 'tuple' || target.k === 'list') target.items.forEach((t, i) => bind(t, v && v.list ? v.list[i] : null, env));
      else if (target.k === 'attr' && target.v.k === 'name' && target.v.v === 'self') { const self = env.get('self'); if (self && self.inst) self.inst.attrs[target.name] = v; }
      else if (target.k === 'attr' && target.name === 'weight' && v && v.param === 'weight') {
        const m = ev(target.v, env);
        if (m && (m.mod === 'linear' || m.mod === 'embed') && (v.of.mod === 'linear' || v.of.mod === 'embed')) { m.tieTo = v.of; tiedSpecs.push(m); }
      }
    }

    const SIMPLE_ACT = { 'nn.SiLU': 'silu', 'nn.ReLU': 'relu', 'nn.GELU': 'gelu', 'nn.Sigmoid': 'sigmoid', 'nn.Tanh': 'tanh', 'nn.Softmax': 'softmax', 'nn.Identity': 'identity' };
    const FN_ACT = {
      'F.silu': 'silu', 'F.relu': 'relu', 'F.gelu': 'gelu', 'F.sigmoid': 'sigmoid', 'F.tanh': 'tanh', 'F.softmax': 'softmax',
      'torch.sigmoid': 'sigmoid', 'torch.tanh': 'tanh', 'torch.relu': 'relu', 'torch.softmax': 'softmax',
      'torch.nn.functional.silu': 'silu', 'torch.nn.functional.relu': 'relu', 'torch.nn.functional.gelu': 'gelu', 'torch.nn.functional.softmax': 'softmax',
    };
    const normPath = p => p.replace(/^torch\.nn\.functional\./, 'F.').replace(/^torch\.nn\./, 'nn.');

    function call(e, env) {
      const fv = ev(e.fn, env);
      const A = e.args.map(a => (a.k === 'star' ? ev(a.v, env) : ev(a, env)));
      const KW = Object.fromEntries(Object.entries(e.kw).map(([k2, v]) => [k2, ev(v, env)]));
      const arg = (i, name, d) => (name in KW ? KW[name] : A.length > i ? A[i] : d);
      // constructor-time specs
      if (fv && fv.path) {
        const p = normPath(fv.path);
        if (p === 'super' || p.endsWith('.__init__')) return null;
        if (p === 'range') {
          const R = A.map(num2); const [a, b, s] = R.length === 1 ? [0, R[0], 1] : [R[0], R[1], R[2] || 1];
          if ([a, b, s].every(x => typeof x === 'number')) { const l = []; for (let x = a; x < b; x += s) l.push(x); return { list: l }; }
          warnings.push(`Couldn't work out range(${e.args.map(x => unparse(x)).join(', ')}); the loop was skipped.`); return { list: [] };
        }
        if (p === 'len') return A[0] && A[0].list ? A[0].list.length : null;
        if (p === 'print' || p === 'isinstance') return null;
        if (p === 'int') return typeof A[0] === 'number' ? Math.trunc(A[0]) : A[0];
        if (p === 'list' || p === 'tuple') return A[0];
        if (p === 'zip') { const n = Math.min(...A.map(x => (x && x.list ? x.list.length : 0))); return { list: Array.from({ length: n }, (_, i) => ({ list: A.map(x => x.list[i]) })) }; }
        if (p === 'enumerate') return { list: (A[0] && A[0].list || []).map((x, i) => ({ list: [i, x] })) };
        if (p === 'nn.Embedding') return { mod: 'embed', num: arg(0, 'num_embeddings'), dim: arg(1, 'embedding_dim') };
        if (p === 'torch.arange' && A[0] && A[0].sizeOf) return { arange: A[0].sizeOf };
        if (p === 'split_heads' && isT(A[0])) return addNode('heads', { heads: dimStr(A[1]) }, [A[0]]);
        if (p === 'merge_heads' && isT(A[0])) return addNode('merge', {}, [A[0]]);
        if (p === 'causal_mask' && isT(A[0])) return addNode('mask', {}, [A[0]]);
        if (p === 'apply_rope' && isT(A[0])) { const b = arg(1, 'base', 10000); return addNode('rope', { base: typeof b === 'number' ? String(+b) : '10000' }, [A[0]]); }
        if ((p === 'apply_rope_axial' || p === 'apply_rope_mrope') && isT(A[0]) && isT(A[1])) {
          const b = arg(2, 'base', 10000), sc = arg(3, 'sections', null);
          return addNode('rope', { base: typeof b === 'number' ? String(+b) : '10000', mode: p === 'apply_rope_axial' ? 'axial' : 'mrope', sections: sc && sc.list ? sc.list.join(', ') : '16, 24, 24' }, [A[0], A[1]]);
        }
        if (p === 'apply_mask' && isT(A[0]) && isT(A[1])) return addNode('mask', { mode: 'given' }, [A[0], A[1]]);
        // a named op from the helpers above: a custom op that keeps its name (its definition is written with the code)
        if (HELPER_OPS[p] && A.some(isT)) {
          const tens = A.filter(isT), extra = e.args.filter((a, i) => !isT(A[i])).map(a => unparse(a, env)).concat(Object.entries(e.kw).map(([a, b]) => a + '=' + unparse(b, env)));
          return addNode('func', { fn: p, extra: extra.join(', '), n: tens.length }, tens);
        }
        if (p.startsWith('nn.init.')) {
          const target = A[0], fn = p.slice(8).replace(/_$/, '');
          if (target && target.param === 'bias' && target.of.mod === 'linear' && fn === 'zeros') { target.of.zero_bias = true; return null; }
          if (target && target.param === 'weight' && target.of.mod === 'embed' && fn === 'normal' && A.length > 1 && !('std' in KW) && A.length < 3) { /* normal_(w, mean) */ }
          if (target && target.param === 'weight' && (target.of.mod === 'linear' || target.of.mod === 'embed')) {
            const m = target.of;
            if (INITS[fn]) {
              m.init = fn;
              const spec = INITS[fn], v = spec.arg ? (spec.arg in KW ? KW[spec.arg] : spec.arg === 'gain' && A.length > 1 ? A[1] : spec.arg === 'std' && A.length > 2 ? A[2] : undefined) : undefined;
              m.init_arg = v !== undefined ? valStr(v) : '';
              if (fn === 'kaiming_uniform' || fn === 'kaiming_normal') m.init_arg = '';
            } else warnings.push(`nn.init.${fn}_ isn't one of the supported initializers; kept the default.`);
          }
          return null;
        }
        if (p === 'nn.Linear') return { mod: 'linear', in: arg(0, 'in_features'), out: arg(1, 'out_features'), bias: arg(2, 'bias', true) };
        if (p === 'nn.LayerNorm' || p === 'nn.RMSNorm') return { mod: 'norm', kind: p.slice(3), dim: arg(0, 'normalized_shape'), eps: KW.eps };
        if (p === 'nn.BatchNorm1d' || p === 'nn.InstanceNorm1d') return { mod: 'norm', kind: p.slice(3), dim: arg(0, 'num_features'), eps: KW.eps };
        if (p === 'nn.GroupNorm') return { mod: 'norm', kind: 'GroupNorm', dim: arg(1, 'num_channels'), groups: arg(0, 'num_groups'), eps: KW.eps };
        if (p === 'nn.Dropout') return { mod: 'dropout', p: arg(0, 'p', 0.5) };
        if (SIMPLE_ACT[p]) return { mod: 'act', fn: SIMPLE_ACT[p] === 'gelu' && KW.approximate && KW.approximate.str === 'tanh' ? 'gelu_tanh' : SIMPLE_ACT[p], dim: arg(0, 'dim', -1) };
        if (p === 'nn.Sequential') return { mod: 'seq', items: A.flatMap(x => (x && x.list ? x.list : [x])) };
        if (p === 'nn.ModuleList') return { list: A[0] && A[0].list ? A[0].list : [] };
        if (classes.has(p)) return instantiate(classes.get(p), A, KW, env);
        if (p.startsWith('nn.') && !A.concat(Object.values(KW)).some(isT)) {
          return { mod: 'custom', init: unparse(e, env) };
        }
        // functional calls in forward
        const tens = A.filter(isT);
        if (FN_ACT[p] && tens.length) return applyAct(FN_ACT[p] === 'gelu' && KW.approximate && KW.approximate.str === 'tanh' ? 'gelu_tanh' : FN_ACT[p], A[0], arg(1, 'dim', -1));
        if ((p === 'F.dropout' || p === 'nn.functional.dropout') && tens.length) return A[0];
        if ((p === 'torch.add' || p === 'torch.mul') && tens.length === 2) return addNode('binop', { op: p === 'torch.add' ? '+' : '*' }, tens);
        if (p === 'torch.cat' || p === 'torch.stack') {
          const parts = (A[0] && A[0].list || []).filter(isT);
          if (parts.length) return addNode('func', { fn: p, extra: 'dim=' + valStr(arg(1, 'dim', 0)), n: parts.length }, parts);
        }
        if (tens.length) {
          const extra = e.args.filter((a, i) => !isT(A[i])).map(a => unparse(a, env)).concat(Object.entries(e.kw).map(([a, b]) => a + '=' + unparse(b, env)));
          warnings.push(`${p}() is kept as a custom op.`);
          return addNode('func', { fn: p, extra: extra.join(', '), n: tens.length }, tens);
        }
        if (p === 'self.apply') { warnings.push('Weights set through self.apply(...) are not read; choose inits on the Linear layers instead.'); return null; }
        if (!fv.missing) warnings.push(`Ignored call to ${p}().`);
        return null;
      }
      if (fv && fv.method) {
        const m = fv.method, self = fv.self;
        if (['sigmoid', 'relu', 'tanh', 'softmax'].includes(m)) return applyAct(m, self, arg(0, 'dim', -1));
        if (m === 'size') return { sizeOf: self, dim: A[0] };
        if (m === 'dim') return null;
        if (m === 'transpose' && ['-2,-1', '-1,-2'].includes(A.map(x => valStr(x)).join(','))) return addNode('transpose', {}, [self]);
        // norm(x.transpose(1, 2)).transpose(1, 2): the channel-first idiom collapses back into one norm layer
        const TR = new Set(['1, 2', '2, 1', '1, -1', '-1, 1']);
        if (m === 'transpose' && TR.has(e.args.map(a => unparse(a, env)).join(', '))) {
          const nn_ = g.nodes.find(z => z.id === self.t);
          const inE = nn_ && nn_.type === 'norm' && NORMS[nn_.params.kind] && NORMS[nn_.params.kind].channelFirst && g.edges.find(z => z.to === nn_.id);
          const tr = inE && g.nodes.find(z => z.id === inE.from);
          if (tr && tr.type === 'func' && tr.params.fn === '.transpose' && TR.has(tr.params.extra) && g.edges.filter(z => z.from === tr.id).length === 1) {
            const src = g.edges.find(z => z.to === tr.id);
            inE.from = src.from;
            g.edges = g.edges.filter(z => z !== src);
            g.nodes = g.nodes.filter(z => z !== tr);
            for (const gr of g.groups) gr.nodes = gr.nodes.filter(id => id !== tr.id);
            const wi = warnings.indexOf('Tensor method .transpose() is kept as a custom op.'); if (wi >= 0) warnings.splice(wi, 1);
            seqInput = true;
            return self;
          }
        }
        const extra = e.args.map(a => unparse(a, env)).concat(Object.entries(e.kw).map(([a, b]) => a + '=' + unparse(b, env)));
        const tens = A.filter(isT);
        warnings.push(`Tensor method .${m}() is kept as a custom op.`);
        return addNode('func', { fn: '.' + m, extra: extra.join(', '), n: 1 + tens.length }, [self, ...tens]);
      }
      // calling a module value
      if (fv && (fv.mod || fv.inst)) return applyMod(fv, A, fv.attrName);
      if (fv && fv.list && A.length) { warnings.push('Called a list; ignored.'); return A[0]; }
      return null;
    }
    function applyAct(fn, x, dim) {
      if (!isT(x)) return null;
      if (fn === 'identity') return x;
      return addNode('act', { fn, dim: String(valStr(dim)) }, [x]);
    }
    function applyMod(m, args, attrName) {
      const x = args[0];
      if (m.inst) return callInstance(m, args, attrName);
      switch (m.mod) {
        case 'embed': {
          const ei = { init: m.init || 'default', init_arg: m.init_arg || '' };
          const r = x && x.arange ? addNode('embed', Object.assign({ num: dimStr(m.num), dim: dimStr(m.dim), positions: true, name: m.attrName || '' }, ei), [x.arange])
            : addNode('embed', Object.assign({ num: dimStr(m.num), dim: dimStr(m.dim), positions: false, name: m.attrName || '' }, ei), [x]);
          m._node = r.t; return r;
        }
        case 'linear': { const r = addNode('linear', { in: dimStr(m.in), out: dimStr(m.out), bias: m.bias !== false, init: m.init || 'default', init_arg: m.init_arg || '', zero_bias: !!m.zero_bias, name: m.attrName || '' }, [x]); m._node = r.t; return r; }
        case 'norm': return addNode('norm', { kind: m.kind, dim: dimStr(m.dim), groups: num2(m.groups) || 8, eps: m.eps != null ? valStr(m.eps) : '', name: m.attrName || '' }, [x]);
        case 'dropout': return addNode('dropout', { p: m.p, name: m.attrName || '' }, [x]);
        case 'act': return applyAct(m.fn, x, m.dim);
        case 'seq': { let v = x; m.items.forEach((s, i) => { if (s && (s.mod || s.inst) && !s.attrName) s.attrName = (m.attrName || 'seq') + '_' + i; v = applyMod(s, [v], s.attrName); }); return v; }
        case 'custom': return addNode('func', { init: m.init, fn: '', n: args.filter(isT).length || 1, name: m.attrName || '' }, args.filter(isT));
      }
      return null;
    }
    function instantiate(cls, A, KW, callerEnv) {
      usedClasses.add(cls.name);
      const init = cls.body.find(s => s.k === 'def' && s.name === '__init__');
      const inst = { cls: cls.name, attrs: {}, def: cls, hints: {}, defaults: {} };
      const val = { inst };
      const env = new Map([['self', val]]);
      if (init) {
        const params = init.params.slice(1);
        params.forEach((p, i) => {
          let v = p.name in KW ? KW[p.name] : i < A.length ? A[i] : p.def ? ev(p.def, env) : undefined;
          if (typeof v === 'number' && Number.isInteger(v) && callerEnv && callerEnv.__nested !== false) {
            v = newSym(p.name, v);
          }
          if (p.def) { try { const dv = ev(p.def, new Map()); if (typeof dv === 'number') inst.defaults[p.name] = dv; } catch {} }
          if (v === undefined) v = newSym(p.name, 64);
          if (v && v.sym && /^[A-Za-z_]\w*$/.test(v.sym) && !(v.sym in inst.hints)) inst.hints[v.sym] = p.name; // remember the class's own argument name
          env.set(p.name, v);
        });
        exec(init.body, env);
      }
      for (const [name, v] of Object.entries(inst.attrs)) tagAttr(v, name);
      return val;
    }
    function tagAttr(v, name) {
      if (!v || typeof v !== 'object') return;
      if (v.mod || v.inst) { if (!v.attrName) v.attrName = name; }
      if (v.list) v.list.forEach((x, i) => tagAttr(x, name + '_' + i));
      if (v.mod === 'seq') v.items.forEach((x, i) => tagAttr(x, name + '_' + i));
    }
    function callInstance(m, args, attrName) {
      const fwd = m.inst.def.body.find(s => s.k === 'def' && s.name === 'forward');
      if (!fwd) { warnings.push(`${m.inst.cls} has no forward().`); return args[0]; }
      const parent = curGroup, top = true;
      {
        const nth = g.groups.filter(x => x.cls === m.inst.cls).length + 1;
        curGroup = { id: uid('g'), parent: parent ? parent.id : null, cls: m.inst.cls, attr: pyIdent(attrName || m.inst.cls.toLowerCase() + nth), label: `${m.inst.cls.replace(/([a-z])([A-Z])/g, '$1 $2')} ${nth}`, nodes: [], argHints: Object.assign({}, m.inst.hints), argDefaults: Object.assign({}, m.inst.defaults) };
        g.groups.push(curGroup);
      }
      const env = new Map([['self', m]]);
      fwd.params.slice(1).forEach((p, i) => env.set(p.name, i < args.length ? args[i] : p.def ? ev(p.def, env) : null));
      const r = exec(fwd.body, env);
      if (top) {
        // a module returning a tuple of tensors becomes a block with several outputs, its tensor arguments its inputs
        const outs = isT(r) ? [r] : r && r.list && r.list.length && r.list.every(isT) ? r.list : null;
        const tin = fwd.params.slice(1).map((p, i) => [p.name, args[i]]).filter(([, v]) => isT(v));
        // the call's tensor arguments in order (and their names) become the block's inputs, its results its outputs
        if (outs && (outs.length > 1 || tin.length > 1)) {
          curGroup.outs = outs.map(v => v.t); curGroup.extIn = tin.map(([, v]) => v.t); curGroup.inNames = tin.map(([nm]) => nm);
          const ret = [...fwd.body].reverse().find(s => s.k === 'return'), rv = ret && ret.value;
          if (rv && rv.k === 'tuple' && rv.items.length === outs.length) curGroup.outNames = rv.items.map(x => (x.k === 'name' ? x.v : null));
        }
        curGroup = parent;
      }
      return r;
    }
    function exec(body, env) {
      for (const s of body) {
        switch (s.k) {
          case 'assign': {
            const before = g.nodes.length, v = ev(s.value, env); s.targets.forEach(t => bind(t, v, env));
            if (isT(v) && g.nodes.length > before && s.targets.length === 1 && s.targets[0].k === 'name') { const n = g.nodes.find(z => z.id === v.t); if (n && 'var' in n.params && !n.params.var && n.type !== 'output') n.params.var = s.targets[0].v; }
            break;
          }
          case 'expr': ev(s.value, env); break;
          case 'return': return s.value ? ev(s.value, env) : null;
          case 'for': {
            const it = ev(s.iter, env);
            const seq = it && it.list ? it.list : [];
            for (const v of seq) { bind(s.target, v, env); const r = exec(s.body, env); if (r !== undefined) return r; }
            break;
          }
          case 'if': {
            const t = ev(s.test, env);
            const r = exec(t === null ? (warnings.push('An if-condition could not be decided; took the first branch.'), s.body) : truthy(t) ? s.body : s.orelse, env);
            if (r !== undefined) return r; break;
          }
          case 'block': { const r = exec(s.body, env); if (r !== undefined) return r; break; }
        }
      }
      return undefined;
    }

    // build root
    const rootInit = root.body.find(s => s.k === 'def' && s.name === '__init__');
    const rootInst = { cls: root.name, attrs: {}, def: root };
    const rootVal = { inst: rootInst };
    const env = new Map([['self', rootVal]]);
    if (rootInit) {
      for (const p of rootInit.params.slice(1)) {
        const d = p.def ? ev(p.def, env) : undefined;
        if (typeof d === 'number' && Number.isInteger(d)) { g.symbols[p.name] = d; env.set(p.name, { sym: p.name }); }
        else if (d === undefined) { g.symbols[p.name] = 64; env.set(p.name, { sym: p.name }); warnings.push(`${p.name} has no default; set to 64.`); }
        else env.set(p.name, d);
        symCount[p.name] = 1;
      }
      exec(rootInit.body, env);
    }
    for (const [name, v] of Object.entries(rootInst.attrs)) tagAttr(v, name);
    const fwd = root.body.find(s => s.k === 'def' && s.name === 'forward');
    const inputs = fwd.params.slice(1).map(p => {
      const n = makeNode('input', { name: p.name, shape: 'e, ?' }); g.nodes.push(n); return { t: n.id };
    });
    const fenv = new Map([['self', rootVal]]);
    fwd.params.slice(1).forEach((p, i) => fenv.set(p.name, inputs[i]));
    let ret = exec(fwd.body, fenv);
    const rets = ret && ret.list ? ret.list : [ret];
    rets.forEach((r, i) => { if (isT(r)) addNode('output', { name: rets.length > 1 ? 'out' + i : 'out' }, [r]); });
    // loss(): rebuild loss nodes from the forms Torch Canvas writes (see lossPlan)
    const lossM = findMethod(src, root.name, 'loss');
    const newTargets = [];
    if (lossM) {
      const outs = g.nodes.filter(n => n.type === 'output');
      lossM.params.slice(0, outs.length).forEach((nm, i) => { outs[i].params.name = nm; });
      const doc = lossM.doc || '';
      let ret = lossM.body.join(' ').replace(/^\s*return\s+/, '').trim();
      if (ret.startsWith('(') && ret.endsWith(')')) ret = ret.slice(1, -1).trim();
      const splitTop = (str, sep) => { const out = []; let d = 0, cur = ''; for (let i = 0; i < str.length; i++) { const ch = str[i]; if ('([{'.includes(ch)) d++; else if (')]}'.includes(ch)) d--; if (d === 0 && str.startsWith(sep, i)) { out.push(cur.trim()); cur = ''; i += sep.length - 1; continue; } cur += ch; } if (cur.trim()) out.push(cur.trim()); return out; };
      const FN = { cross_entropy: 'cross_entropy', binary_cross_entropy_with_logits: 'bce', mse_loss: 'mse', l1_loss: 'l1', huber_loss: 'huber', smooth_l1_loss: 'huber' };
      for (const term of splitTop(ret, '+')) {
        const m = /^(?:([\d.eE+-]+)\s*\*\s*)?F\.(\w+)\((.*)\)$/.exec(term);
        if (!m || !FN[m[2]]) { warnings.push(`loss(): couldn't read "${term.slice(0, 60)}"; only the forms Torch Canvas writes are read back.`); continue; }
        const parts = splitTop(m[3], ','), pos = parts.filter(a => !/^\w+\s*=/.test(a)), kw = Object.fromEntries(parts.filter(a => /^\w+\s*=/.test(a)).map(a => { const i = a.indexOf('='); return [a.slice(0, i).trim(), a.slice(i + 1).trim()]; }));
        const pm = /^(\w+)(\[:, :-1\])?(\.flatten\(0, -2\))?$/.exec(pos[0] || ''), tm = /^(\w+)(\[:, 1:\])?(\.flatten\(\))?(\.float\(\))?$/.exec(pos[1] || '');
        if (!pm || !tm) { warnings.push(`loss(): couldn't read the arguments of ${m[2]}.`); continue; }
        const o = outs.find(z => z.params.name === pm[1]);
        if (!o) { warnings.push(`loss(): ${pm[1]} isn't an output of forward().`); continue; }
        const fn = FN[m[2]], p = { fn, weight: m[1] ? String(+m[1]) : '1' };
        let tname = tm[1];
        if (fn === 'cross_entropy') {
          if (kw.label_smoothing) p.label_smoothing = String(+kw.label_smoothing);
          const md = /labels come from (\w+), masking ([\d.]+)% of them(?: with id (\d+))?/.exec(doc);
          if (pm[2] && tm[2]) p.mode = 'next';
          else if (tname === 'labels' && kw.ignore_index === '-100' && md) { p.mode = 'masked'; tname = md[1]; p.mask_prob = String(+md[2] / 100); p.mask_id = md[3] || ''; }
          else { p.mode = 'given'; if (kw.ignore_index != null) p.ignore_index = String(+kw.ignore_index); }
        }
        if (fn === 'huber' && kw.delta) p.delta = String(+kw.delta);
        let tn = g.nodes.find(z => z.type === 'input' && z.params.name === tname);
        if (!tn) { tn = makeNode('input', { name: tname, shape: 'e, ?' }); g.nodes.push(tn); newTargets.push({ n: tn, fn, out: o }); }
        const src = g.edges.find(e => e.to === o.id).from;
        addNode('loss', p, [{ t: src }, { t: tn.id }]);
      }
    }

    // rename h → h1 when h2 exists
    const ren = new Map();
    for (const [base, c] of Object.entries(symCount)) if (c > 1 && g.symbols[base] != null && !g.symbols[base + '1']) ren.set(base, base + '1');
    if (ren.size) {
      for (const [a, b] of ren) { g.symbols[b] = g.symbols[a]; delete g.symbols[a]; }
      for (const n of g.nodes) for (const f of ['in', 'out', 'dim', 'init', 'num', 'heads']) if (typeof n.params[f] === 'string') n.params[f] = n.params[f].replace(IDENT, t => ren.get(t) || t);
      // block uses' sizes and the module calls' name hints follow the renamed sizes
      for (const n of g.nodes) if (n.type === 'block' && n.params.args) for (const k of Object.keys(n.params.args)) n.params.args[k] = String(n.params.args[k]).replace(IDENT, t => ren.get(t) || t);
      for (const gr of g.groups || []) if (gr.argHints) gr.argHints = Object.fromEntries(Object.entries(gr.argHints).map(([k, v]) => [ren.get(k) || k, v]));
    }
    // infer input shapes from first consumer
    const ix = index(g);
    for (const n of g.nodes.filter(n => n.type === 'input')) {
      const seen = new Set(); let dim = null;
      const walk = id => { if (dim || seen.has(id)) return; seen.add(id); for (const o of ix.outs.get(id)) { const c = ix.byId.get(o.to); if (c.type === 'linear') { dim = c.params.in; return; } if (c.type === 'norm') { dim = c.params.dim; return; } walk(c.id); } };
      const feedsEmbed = ix.outs.get(n.id).some(o => ix.byId.get(o.to).type === 'embed');
      if (feedsEmbed) { n.params.shape = 'e, t'; if (g.symbols.t == null) g.symbols.t = 64; continue; }
      walk(n.id); n.params.shape = (seqInput ? 'e, t, ' : 'e, ') + (dim || 'd_in');
      if (seqInput && g.symbols.t == null) g.symbols.t = 16;
    }
    // the __main__ test call, when there is one, gives each input's real shape: model(torch.randn(2, 1024, 1176), …)
    {
      const m = /\n\s*\w+\s*=\s*model\(([^\n]*)\)\s*\n/.exec(src.slice(src.indexOf('__main__')));
      const split = t => { const out = []; let dep = 0, cur = ''; for (const ch of t) { if ('([{'.includes(ch)) dep++; if (')]}'.includes(ch)) dep--; if (ch === ',' && !dep) { out.push(cur.trim()); cur = ''; } else cur += ch; } if (cur.trim()) out.push(cur.trim()); return out; };
      const dimsOf = a => { let r = /^torch\.(?:randn|rand|zeros|ones|empty)\((.*)\)$/.exec(a); if (r) return split(r[1]).filter(x => !x.includes('=')).map(Number);
        r = /^torch\.randint\((.*)\)$/.exec(a); if (r) { const p = split(r[1]), tup = p.find(x => x.startsWith('(')); return tup ? split(tup.slice(1, -1)).map(Number) : null; } return null; };
      const ins = g.nodes.filter(n => n.type === 'input' && !newTargets.some(x => x.n === n)), args = m ? split(m[1]) : [];
      // per-input example lines: name = expression  # (n=512, c_in=1176) — the shape with its size names, and the example
      const main = src.slice(src.indexOf('__main__')), byName = new Map(ins.map(n => [pyIdent(n.params.name || 'x'), n]));
      for (const mm of main.matchAll(/^ {4}(\w+) = (.+?)  # \((.*)\)$/gm)) {
        const n = byName.get(mm[1]); if (!n) continue;
        const dims = mm[3].split(',').map(x => x.trim()).filter(Boolean).map(d => { const kv = /^([A-Za-z_]\w*)=(-?\d+)$/.exec(d); if (kv) { g.symbols[kv[1]] = +kv[2]; return kv[1]; } return d; });
        n.params.shape = dims.join(', '); n.params.demo = mm[2]; n.params.demoRead = true;
      }
      const read = ins.filter(n => n.params.demoRead);
      if (m && args.length === ins.length) ins.forEach((n, i) => {
        if (n.params.demoRead) return;
        const dims = dimsOf(args[i]); if (!dims || dims.some(x => !Number.isInteger(x))) return;
        const cur = parseShape(n.params.shape), feeds = ix.outs.get(n.id).some(o => ix.byId.get(o.to).type === 'embed');
        if (dims.length === cur.length && (feeds || dims.length < 3)) return; // the guess already has this shape
        // batch first, then sequence-like sizes (t, t2, …), then the width the first layer reads
        const sh = [BATCH], mids = feeds ? dims.slice(1) : dims.slice(1, -1);
        for (const v of mids) { let nm = 't', k = 2; while (g.symbols[nm] != null && g.symbols[nm] !== v && !sh.includes(nm)) nm = 't' + k++; if (sh.includes(nm)) nm = 't' + k++; g.symbols[nm] = v; sh.push(nm); }
        if (!feeds) sh.push(cur[cur.length - 1]);
        if (g.symbols[BATCH] == null || g.symbols[BATCH] === 8) g.symbols[BATCH] = dims[0];
        n.params.shape = sh.join(', ');
      });
      read.forEach(n => delete n.params.demoRead);
    }
    g.nodes = g.nodes.filter(n => n.type === 'input' || n.type === 'output' || ix.outs.get(n.id).length || true);
    // targets that only a loss reads: shaped like what the loss expects
    if (newTargets.length) {
      const sh = inferShapes(g).shapes;
      for (const { n, fn, out } of newTargets) { const P = sh.get(g.edges.find(e => e.to === out.id).from); if (P) n.params.shape = (fn === 'cross_entropy' ? P.slice(0, -1) : P).join(', '); }
    }
    for (const m of tiedSpecs) {
      const n = m._node && g.nodes.find(z => z.id === m._node), o = m.tieTo._node;
      if (n && o && o !== n.id) n.params.tie = o;
      else warnings.push('A weight tie refers to a layer that forward() never uses; it was left out.');
    }
    linkSizes(g);
    groupsToBlocks(g);
    autoLayout(g);
    // code style of the source: named sizes if the top-level model takes size arguments
    const literalDims = !(rootInit && rootInit.params.length > 1);
    return { graph: g, warnings: [...new Set(warnings)], usedClasses: [...usedClasses], literalDims };
  }

  // ── torch.fx JSON importer (from export_graph.py) ──
  function importFx(json) {
    const data = typeof json === 'string' ? JSON.parse(json) : json;
    const g = { name: data.name || 'Model', symbols: { [BATCH]: 8 }, nodes: [], edges: [], groups: [] };
    const warnings = [], idOf = new Map(), groups = new Map();
    const dimSyms = new Map();
    const dim = v => { if (typeof v !== 'number') return String(v); if (!dimSyms.has(v)) { let nm = 'd' + v; g.symbols[nm] = v; dimSyms.set(v, nm); } return dimSyms.get(v); };
    const FN = { 'torch.sigmoid': 'sigmoid', 'torch.tanh': 'tanh', 'torch.relu': 'relu', 'torch.nn.functional.silu': 'silu', 'torch.nn.functional.relu': 'relu', 'torch.nn.functional.gelu': 'gelu', 'torch.nn.functional.softmax': 'softmax', 'torch.softmax': 'softmax', 'torch.nn.functional.sigmoid': 'sigmoid' };
    const BIN = { 'operator.add': '+', 'operator.mul': '*', 'operator.sub': '-', 'operator.truediv': '/', 'operator.matmul': '@', 'torch.add': '+', 'torch.mul': '*' };
    const ACTM = { SiLU: 'silu', ReLU: 'relu', GELU: 'gelu', Sigmoid: 'sigmoid', Tanh: 'tanh', Softmax: 'softmax' };
    for (const fn of data.nodes) {
      const tensorArgs = (fn.args || []).filter(a => a && a.node).map(a => a.node);
      const consts = (fn.args || []).filter(a => !(a && a.node));
      let n = null;
      if (fn.op === 'placeholder') n = makeNode('input', { name: fn.name, shape: fn.shape ? 'e, ' + fn.shape.slice(1).map(dim).join(', ') : 'e, ?' });
      else if (fn.op === 'output') { tensorArgs.forEach((src, i) => { const o = makeNode('output', { name: tensorArgs.length > 1 ? 'out' + i : 'out' }); g.nodes.push(o); g.edges.push({ id: uid('e'), from: idOf.get(src), to: o.id, port: 0 }); }); continue; }
      else if (fn.op === 'call_module') {
        const m = fn.module || {}; const attr = fn.target.split('.').pop();
        if (m.type === 'Linear') n = makeNode('linear', { in: dim(m.in_features), out: dim(m.out_features), bias: !!m.bias, name: attr });
        else if (NORMS[m.type]) n = makeNode('norm', { kind: m.type, dim: dim(m.dim), groups: m.num_groups || 8, name: attr });
        else if (m.type === 'Dropout') n = makeNode('dropout', { p: m.p ?? 0.1, name: attr });
        else if (ACTM[m.type]) n = makeNode('act', { fn: ACTM[m.type], dim: String(m.dim ?? -1) });
        else { n = makeNode('func', { init: m.repr || `nn.${m.type}()`, n: Math.max(1, tensorArgs.length), name: attr }); warnings.push(`${m.type} kept as a custom module.`); }
      } else if (fn.op === 'call_function' || fn.op === 'call_method') {
        const t = fn.target;
        if (BIN[t]) n = tensorArgs.length === 2 ? makeNode('binop', { op: BIN[t] }) : makeNode('binop', { op: BIN[t], const: String(consts[0]) });
        else if (FN[t] || (fn.op === 'call_method' && ['sigmoid', 'relu', 'tanh', 'softmax'].includes(t))) n = makeNode('act', { fn: FN[t] || t, dim: String((fn.kwargs && fn.kwargs.dim) ?? -1) });
        else { n = makeNode('func', { fn: fn.op === 'call_method' ? '.' + t : t, extra: consts.map(String).join(', '), n: Math.max(1, tensorArgs.length) }); warnings.push(`${t} kept as a custom op.`); }
      } else continue;
      g.nodes.push(n); idOf.set(fn.name, n.id);
      tensorArgs.forEach((src, port) => { if (idOf.has(src)) g.edges.push({ id: uid('e'), from: idOf.get(src), to: n.id, port }); });
      if (fn.group) {
        if (!groups.has(fn.group.path)) { const gr = { id: uid('g'), cls: fn.group.cls, attr: pyIdent(fn.group.path.replace(/\./g, '_')), label: fn.group.path, nodes: [] }; groups.set(fn.group.path, gr); g.groups.push(gr); }
        groups.get(fn.group.path).nodes.push(n.id);
      }
    }
    groupsToBlocks(g);
    autoLayout(g);
    return { graph: g, warnings: [...new Set(warnings)] };
  }

  // ── auto layout: clusters (groups) as columns, flowing bottom → top ──
  function nodeSize(n) {
    if (isCircle(n)) return { w: 26, h: 26 };
    if (n.type === 'act') return { w: 44, h: 32 };
    if (n.type === 'block') return { w: 140, h: 50 };
    if (n.type === 'input' || n.type === 'output') return { w: 90, h: 30 };
    return { w: 104, h: 40 };
  }
  function autoLayout(g, sz = nodeSize) {
    const ix = index(g), { order } = topo(g, ix);
    const pos = new Map(order.map((id, i) => [id, i]));
    const groupOrder = new Map();
    (g.groups || []).forEach(gr => { const m = Math.min(...gr.nodes.map(id => pos.get(id) ?? 1e9)); groupOrder.set(gr.id, m); });
    const sortedGroups = [...(g.groups || [])].sort((a, b) => groupOrder.get(a.id) - groupOrder.get(b.id));
    const gIdx = new Map(sortedGroups.map((gr, i) => [gr.id, i]));
    const clusterOf = new Map(), after = new Map();
    for (const id of order) {
      const gr = ix.groupOf.get(id);
      if (gr) { clusterOf.set(id, 'g' + gIdx.get(gr.id)); after.set(id, gIdx.get(gr.id)); continue; }
      let k = -1; for (const s of ix.ins.get(id)) if (s && after.has(s)) k = Math.max(k, after.get(s));
      after.set(id, k); clusterOf.set(id, 'f' + k);
    }
    const keys = ['f-1']; sortedGroups.forEach((_, i) => keys.push('g' + i, 'f' + i));
    const LAYER = 74, GAP_X = 26, COL_GAP = 70, BASE = 600;
    let cx = 60;
    for (const key of keys) {
      const ids = order.filter(id => clusterOf.get(id) === key);
      if (!ids.length) continue;
      const layer = new Map();
      for (const id of ids) { let l = 0; for (const s of ix.ins.get(id)) if (s && layer.has(s)) l = Math.max(l, layer.get(s) + 1); layer.set(id, l); }
      const layers = [];
      for (const id of ids) (layers[layer.get(id)] = layers[layer.get(id)] || []).push(id);
      let width = 0; const placed = new Map();
      const rows = layers.map((row, li) => {
        row.sort((a, b) => bary(a) - bary(b));
        function bary(id) { const ps = ix.ins.get(id).filter(s => placed.has(s)); return ps.length ? ps.reduce((t, s) => t + placed.get(s), 0) / ps.length : 0; }
        let x = 0; const xs = row.map(id => { const w = sz(ix.byId.get(id)).w; const c = x + w / 2; x += w + GAP_X; return c; });
        const rw = x - GAP_X; width = Math.max(width, rw);
        row.forEach((id, i) => placed.set(id, xs[i]));
        return { row, xs, rw, li };
      });
      // long stacks wrap into a snake of columns (up, then down, then up …) instead of one very tall column
      const CH = 10, wrap = rows.length > 12, ncols = wrap ? Math.ceil(rows.length / CH) : 1;
      for (const r of rows) {
        const off = (width - r.rw) / 2;
        const col = wrap ? Math.floor(r.li / CH) : 0, k = wrap ? r.li % CH : r.li;
        const y = wrap && col % 2 === 1 ? BASE - (CH - 1 - k) * LAYER : BASE - k * LAYER;
        r.row.forEach((id, i) => {
          const n = ix.byId.get(id);
          n.x = cx + col * (width + COL_GAP) + r.xs[i] + off; n.y = y;
        });
      }
      cx += ncols * (width + COL_GAP);
    }
    return g;
  }

  // ── preset: the residual MLP from the reference diagram ──
  // Same network, with the gated residual layer defined once as a block and repeated three times.
  function presetResidualMLPBlocks() {
    const bg = { name: 'GatedBlock', symbols: { e: 8, d: 256, h: 512 }, nodes: [], edges: [], groups: [] };
    const B = (type, params, x, y) => { const n = makeNode(type, params, x, y); bg.nodes.push(n); return n; };
    const BE = (a, b, port = 0) => bg.edges.push({ id: uid('e'), from: a.id, to: b.id, port });
    const bx = B('input', { name: 'x', shape: 'e, d' }, 60, 650);
    const norm = B('norm', { kind: 'RMSNorm', dim: 'd', name: 'norm' }, 200, 600);
    const w = B('linear', { in: 'd', out: 'h', name: 'w' }, 146, 515);
    const v = B('linear', { in: 'd', out: 'h', name: 'v' }, 258, 515);
    const sg = B('act', { fn: 'sigmoid' }, 100, 430);
    const m1 = B('binop', { op: '*' }, 146, 380);
    const m2 = B('binop', { op: '*' }, 200, 325);
    const wo = B('linear', { in: 'h', out: 'd', name: 'w_out' }, 200, 270);
    const add = B('binop', { op: '+' }, 200, 205);
    const bo = B('output', { name: 'out' }, 200, 140);
    BE(bx, norm); BE(norm, w); BE(norm, v); BE(w, sg); BE(sg, m1, 0); BE(w, m1, 1); BE(m1, m2, 0); BE(v, m2, 1); BE(m2, wo); BE(bx, add, 0); BE(wo, add, 1); BE(add, bo);
    const blk = { id: uid('b'), name: 'GatedBlock', graph: bg, labels: { h: 'hidden' } };

    const g = { name: 'ResidualMLP', symbols: { e: 8, d_in: 784, f: 256, h1: 512, h_o: 256, o: 10 }, nodes: [], edges: [], groups: [], blocks: { [blk.id]: blk } };
    const N = (type, params, x, y) => { const n = makeNode(type, params, x, y); g.nodes.push(n); return n; };
    const E = (a, b, port = 0) => g.edges.push({ id: uid('e'), from: a.id, to: b.id, port });
    const x = N('input', { name: 'x', shape: 'e, d_in' }, 90, 300);
    const down = N('linear', { in: 'd_in', out: 'f', name: 'down', var: 'x' }, 230, 300);
    E(x, down);
    const blocks = N('block', { block: blk.id, args: { h: 'h1' }, nIn: 1, repeat: 3, name: 'blocks' }, 450, 300);
    E(down, blocks);
    const prev = blocks;
    const hx = 680;
    const fn = N('norm', { kind: 'RMSNorm', dim: 'f', name: 'final_norm' }, hx, 300);
    const hw = N('linear', { in: 'f', out: 'h_o', name: 'head_w' }, hx - 56, 380);
    const hv = N('linear', { in: 'f', out: 'h_o', name: 'head_v' }, hx + 56, 380);
    const hs = N('act', { fn: 'sigmoid' }, hx, 445);
    const hm1 = N('binop', { op: '*' }, hx - 56, 485);
    const hm2 = N('binop', { op: '*' }, hx, 535);
    const op = N('linear', { in: 'h_o', out: 'o', name: 'out_proj' }, hx, 600);
    const sm = N('act', { fn: 'softmax', dim: '-1' }, hx, 665);
    const out = N('output', { name: 'p_hat' }, hx, 730);
    E(prev, fn); E(fn, hw); E(fn, hv); E(hw, hs); E(hs, hm1, 0); E(hw, hm1, 1); E(hm1, hm2, 0); E(hv, hm2, 1); E(hm2, op); E(op, sm); E(sm, out);
    return g;
  }

  function presetResidualMLP() {
    const g = { name: 'ResidualMLP', symbols: { e: 8, d_in: 784, f: 256, h1: 512, h2: 512, h3: 512, h_o: 256, o: 10 }, nodes: [], edges: [], groups: [] };
    const N = (type, params, x, y) => { const n = makeNode(type, params, x, y); g.nodes.push(n); return n; };
    const E = (a, b, port = 0) => g.edges.push({ id: uid('e'), from: a.id, to: b.id, port });
    const x = N('input', { name: 'x', shape: 'e, d_in' }, 130, 690);
    const down = N('linear', { in: 'd_in', out: 'f', name: 'down', var: 'x' }, 130, 590);
    E(x, down);
    let stream = down;
    for (let k = 1; k <= 3; k++) {
      const ox = 300 + (k - 1) * 290, h = 'h' + k;
      const norm = N('norm', { kind: 'RMSNorm', dim: 'f', name: 'norm' + k }, ox + 120, 600);
      const w = N('linear', { in: 'f', out: h, name: 'w' + k }, ox + 66, 515);
      const v = N('linear', { in: 'f', out: h, name: 'v' + k }, ox + 178, 515);
      const sg = N('act', { fn: 'sigmoid' }, ox + 20, 430);
      const m1 = N('binop', { op: '*' }, ox + 66, 380);
      const m2 = N('binop', { op: '*' }, ox + 120, 325);
      const wo = N('linear', { in: h, out: 'f', name: 'w_out' + k }, ox + 120, 270);
      const add = N('binop', { op: '+' }, ox + 120, 205);
      E(stream, norm); E(norm, w); E(norm, v); E(w, sg); E(sg, m1, 0); E(w, m1, 1); E(m1, m2, 0); E(v, m2, 1); E(m2, wo); E(stream, add, 0); E(wo, add, 1);
      stream = add;
    }
    const hx = 1190;
    const fn = N('norm', { kind: 'RMSNorm', dim: 'f', name: 'final_norm' }, hx + 40, 270);
    const hw = N('linear', { in: 'f', out: 'h_o', name: 'head_w' }, hx, 340);
    const hv = N('linear', { in: 'f', out: 'h_o', name: 'head_v' }, hx + 112, 340);
    const hs = N('act', { fn: 'sigmoid' }, hx + 58, 405);
    const hm1 = N('binop', { op: '*' }, hx, 445);
    const hm2 = N('binop', { op: '*' }, hx + 56, 495);
    const op = N('linear', { in: 'h_o', out: 'o', name: 'out_proj' }, hx + 56, 560);
    const sm = N('act', { fn: 'softmax', dim: '-1' }, hx + 56, 625);
    const out = N('output', { name: 'p_hat' }, hx + 56, 690);
    E(stream, fn); E(fn, hw); E(fn, hv); E(hw, hs); E(hs, hm1, 0); E(hw, hm1, 1); E(hm1, hm2, 0); E(hv, hm2, 1); E(hm2, op); E(op, sm); E(sm, out);
    return g;
  }

  // Put generated classes back into an existing Python file: the model's class
  // blocks are replaced in place; everything else in the file is kept.
  function mergePlain(src, gen, used) {
    const lines = src.replace(/\r\n/g, '\n').split('\n');
    const usedSet = new Set(used || []);
    const blocks = [];
    for (let i = 0; i < lines.length; i++) {
      const m = /^class\s+(\w+)/.exec(lines[i]); if (!m || !usedSet.has(m[1])) continue;
      let a = i; while (a > 0 && /^@/.test(lines[a - 1])) a--;
      let j = i + 1; while (j < lines.length && (lines[j].trim() === '' || /^\s/.test(lines[j]))) j++;
      let b = j; while (b > i + 1 && lines[b - 1].trim() === '') b--;
      blocks.push({ name: m[1], a, b }); i = j - 1;
    }
    if (!blocks.length) return null;
    const gl = gen.split('\n');
    const c0 = gl.findIndex(l => /^class /.test(l));
    let c1 = gl.findIndex(l => /^if __name__/.test(l)); if (c1 < 0) c1 = gl.length;
    let classes = gl.slice(c0, c1).join('\n').trim();
    const helperDefs = gl.slice(0, c0).join('\n').split(/\n(?=def )/).filter(b => /^def (\w+)/.test(b.trim())).map(b => b.trim());
    const missing = helperDefs.filter(b => !new RegExp('^def ' + /^def (\w+)/.exec(b)[1] + '\\(', 'm').test(src));
    if (missing.length) classes = missing.join('\n\n\n') + '\n\n\n' + classes;
    const out = [];
    let k = 0;
    for (let i = 0; i < lines.length; i++) {
      const bl = blocks[k];
      if (bl && i === bl.a) {
        if (k === 0) out.push(...classes.split('\n'));
        i = bl.b - 1; k++;
        // swallow blank lines between consecutive removed blocks
        if (k < blocks.length) { let j = i + 1; while (j < blocks[k].a && lines[j].trim() === '') j++; if (j === blocks[k].a) i = j - 1; }
        continue;
      }
      out.push(lines[i]);
    }
    let text = out.join('\n');
    const need = [];
    if (/\btorch\./.test(classes) && !/^import torch\s*$/m.test(text)) need.push('import torch');
    if (/\bnn\./.test(classes) && !/^(import torch\.nn as nn|from torch import nn)\b/m.test(text)) need.push('import torch.nn as nn');
    if (/\bF\./.test(classes) && !/^import torch\.nn\.functional as F\b/m.test(text)) need.push('import torch.nn.functional as F');
    if (need.length) {
      const tl = text.split('\n'); let at = -1;
      for (let i = 0; i < tl.length; i++) { if (/^(import|from)\s/.test(tl[i])) at = i; else if (/^(class|def|if|@)/.test(tl[i])) break; }
      tl.splice(at + 1, 0, ...need); text = tl.join('\n');
    }
    return text.replace(/\n{4,}/g, '\n\n\n');
  }

  // ── marimo notebooks ──
  // marimo notebooks keep imports in `with app.setup:` and model classes as
  // top-level `@app.class_definition` cells. The plain merge drops those
  // decorators and inserts imports at column 0, which marimo rejects.
  const isMarimo = src => /^import marimo\b/m.test(src) && /^\w+\s*=\s*marimo\.App\(/m.test(src);
  const IMPORTS = [
    ['import torch', /^[ \t]*import torch[ \t]*$/m, /\btorch\./],
    ['import torch.nn as nn', /^[ \t]*(import torch\.nn as nn|from torch import nn)\b/m, /\bnn\./],
    ['import torch.nn.functional as F', /^[ \t]*import torch\.nn\.functional as F\b/m, /\bF\./],
  ];
  function mergeMarimo(src, gen, used) {
    const out = mergePlain(src, gen, used);
    if (out == null) return null;
    const app = /^(\w+)\s*=\s*marimo\.App\(/m.exec(src)[1];
    const before = new Set(src.replace(/\r\n/g, '\n').split('\n'));
    const lines = out.split('\n');
    // Imports the plain merge added at column 0 are moved into the setup cell below.
    const added = new Set(IMPORTS.map(i => i[0]).filter(l => lines.includes(l) && !before.has(l)));
    const body = lines.filter(l => !added.has(l));
    // Decorate every regenerated class and helper that is new to the file.
    const seen = new Set(src.match(/^def \w+/gm) || []);
    const res = [];
    for (const l of body) {
      const prev = res.length ? res[res.length - 1] : '';
      const decorated = /^@/.test(prev);
      const cls = /^class\s+(\w+)/.exec(l);
      if (cls && !decorated) res.push(`@${app}.class_definition`);
      else if (/^def \w+/.test(l) && !decorated && !seen.has(/^def \w+/.exec(l)[0]) && !before.has(l)) res.push(`@${app}.function`);
      res.push(l);
    }
    // Only code decides which imports are needed, so the import lines themselves are skipped.
    const code = res.filter(l => !/^\s*(import|from)\s/.test(l)).join('\n');
    return ensureImports(res.join('\n'), code);
  }

  // Put generated classes back into an existing Python file, plain or marimo.
  function mergeIntoSource(src, gen, used) {
    return (isMarimo(src) ? mergeMarimo : mergePlain)(src, gen, used);
  }

  // ── source sync: apply diagram edits to the real source without regenerating it ──
  // The file text stays the source of truth. The diagram edit is the difference between
  // the graph imported from that text and the current graph, measured on generated code.
  // Only the constructor statements and methods that difference touches are rewritten;
  // docstrings, comments, extra methods, and unrelated code stay byte for byte.
  const indentOf = l => /^[ \t]*/.exec(l)[0].length;
  const isBlank = l => l.trim() === '';
  const isComment = l => l.trim().startsWith('#');
  // For each line: does a new logical line (statement) begin here?
  function logicalStarts(lines) {
    const starts = []; let depth = 0, tri = null, cont = false;
    for (const s of lines) {
      starts.push(depth === 0 && !tri && !cont);
      cont = false;
      let i = 0;
      while (i < s.length) {
        if (tri) { const j = s.indexOf(tri, i); if (j < 0) i = s.length; else { tri = null; i = j + 3; } continue; }
        const c = s[i];
        if (c === '#') break;
        if (c === '"' || c === "'") {
          if (s.startsWith(c.repeat(3), i)) { tri = c.repeat(3); i += 3; continue; }
          let j = i + 1; while (j < s.length && s[j] !== c) j += s[j] === '\\' ? 2 : 1;
          i = j + 1; continue;
        }
        if ('([{'.includes(c)) depth++; else if (')]}'.includes(c)) depth = Math.max(0, depth - 1);
        i++;
      }
      if (!tri && depth === 0 && /\\\s*$/.test(s)) cont = true;
    }
    return starts;
  }
  // Statements at one indent. a: first line including leading comments and decorators,
  // p: first code line, b: end (exclusive, trailing blank lines excluded).
  function itemsAt(lines, from, to, indent) {
    const ls = lines.slice(from, to), st = logicalStarts(ls), ps = [];
    for (let i = 0; i < ls.length; i++) if (st[i] && !isBlank(ls[i]) && !isComment(ls[i]) && indentOf(ls[i]) === indent) ps.push(i);
    const items = [];
    for (let k = 0; k < ps.length; k++) {
      let a = ps[k]; const lo = k ? items[k - 1].b - from : 0;
      while (a - 1 >= lo && isComment(ls[a - 1]) && indentOf(ls[a - 1]) === indent) a--;
      let b = k + 1 < ps.length ? ps[k + 1] : ls.length;
      if (k + 1 < ps.length) while (b - 1 > ps[k] && (isBlank(ls[b - 1]) || (isComment(ls[b - 1]) && indentOf(ls[b - 1]) === indent))) b--;
      else while (b - 1 > ps[k] && isBlank(ls[b - 1])) b--;
      items.push({ a: a + from, p: ps[k] + from, b: b + from, text: ls[ps[k]].trim() });
    }
    // a decorator belongs to the statement that follows it
    const out = [];
    for (let k = 0; k < items.length; k++) {
      if (items[k].text.startsWith('@') && k + 1 < items.length) { items[k + 1].a = items[k].a; items[k + 1].deco = true; continue; }
      out.push(items[k]);
    }
    return out;
  }
  function classBlocks(lines) {
    const blocks = [];
    for (let i = 0; i < lines.length; i++) {
      const m = /^class\s+(\w+)/.exec(lines[i]); if (!m) continue;
      let a = i; while (a > 0 && /^@/.test(lines[a - 1])) a--;
      let j = i + 1; while (j < lines.length && (isBlank(lines[j]) || /^\s/.test(lines[j]))) j++;
      let b = j; while (b > i + 1 && isBlank(lines[b - 1])) b--;
      blocks.push({ name: m[1], a, h: i, b }); i = j - 1;
    }
    return blocks;
  }
  // A class as {lines, indent, methods:[{name, a, b, ...}]}, line numbers relative to the block.
  function classParts(lines) {
    const h = lines.findIndex(l => /^class\s/.test(l));
    let i = h + 1; while (i < lines.length && (isBlank(lines[i]) || isComment(lines[i]))) i++;
    const indent = i < lines.length ? indentOf(lines[i]) : 4;
    const methods = itemsAt(lines, h + 1, lines.length, indent).map(it => {
      const m = /^(?:async\s+)?def\s+(\w+)/.exec(it.text); return m ? Object.assign(it, { name: m[1] }) : null;
    }).filter(Boolean);
    return { lines, indent, methods, h };
  }
  // The def header spans p up to the line before the first body statement.
  function methodParts(lines, it, indent) {
    const st = logicalStarts(lines.slice(it.p, it.b));
    let hb = it.p + 1; while (hb < it.b && !st[hb - it.p]) hb++;
    const body = itemsAt(lines, hb, it.b, indentOf(lines[hb] || ''));
    const docstring = body.length && /^[rRbBuU]*("""|'''|"|')/.test(body[0].text) ? body[0] : null;
    return { head: [it.p, hb], bodyFrom: hb, body, docstring, indent: hb < it.b ? indentOf(lines[hb]) : indent + 4 };
  }
  // ── signature parameters ──
  function splitTop(text, ch) {
    const parts = []; let depth = 0, q = null, last = 0;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '\\') i++; else if (c === q) q = null; continue; }
      if (c === '"' || c === "'") q = c;
      else if ('([{'.includes(c)) depth++; else if (')]}'.includes(c)) depth--;
      else if (c === ch && depth === 0) { parts.push(text.slice(last, i)); last = i + 1; }
    }
    parts.push(text.slice(last)); return parts;
  }
  function paramsOf(header) {
    const open = header.indexOf('('); let depth = 0, close = -1, q = null;
    for (let i = open; i < header.length; i++) {
      const c = header[i];
      if (q) { if (c === '\\') i++; else if (c === q) q = null; continue; }
      if (c === '"' || c === "'") q = c; else if ('([{'.includes(c)) depth++; else if (')]}'.includes(c) && --depth === 0) { close = i; break; }
    }
    if (open < 0 || close < 0) return null;
    const segs = splitTop(header.slice(open + 1, close), ',');
    const params = segs.map(raw => {
      const eq = splitTop(raw, '=').length > 1 ? splitTop(raw, '=')[0].length : -1;
      const nm = /^\s*\*{0,2}(\w+)/.exec(raw);
      return { raw, name: nm ? nm[1] : null, eq, def: eq >= 0 ? raw.slice(eq + 1).trim() : null };
    });
    return { before: header.slice(0, open + 1), after: header.slice(close), params };
  }
  function patchParams(orig, base, cur) {
    const o = paramsOf(orig), b = paramsOf(base), c = paramsOf(cur);
    if (!o || !b || !c) return orig;
    const bm = new Map(b.params.filter(p => p.name).map(p => [p.name, p])), cm = new Map(c.params.filter(p => p.name).map(p => [p.name, p]));
    const segs = o.params.slice();
    const find = n => segs.findIndex(p => p.name === n);
    for (const p of c.params) {
      if (!p.name) continue;
      const bp = bm.get(p.name), i = find(p.name);
      if (!bp) { if (i < 0) { const at = segs.length && segs[segs.length - 1].raw.trim() === '' ? segs.length - 1 : segs.length; segs.splice(at, 0, { raw: (at ? ' ' : '') + p.raw.trim(), name: p.name, eq: -1, def: null }); } }
      else if (bp.def !== p.def && i >= 0 && segs[i].eq >= 0 && p.def != null) {
        const s = segs[i], tail = s.raw.slice(s.eq + 1), lead = /^\s*/.exec(tail)[0], trail = /\s*$/.exec(tail)[0];
        segs[i] = Object.assign({}, s, { raw: s.raw.slice(0, s.eq + 1) + lead + p.def + trail, def: p.def });
      }
    }
    for (const p of b.params) if (p.name && !cm.has(p.name)) { const i = find(p.name); if (i >= 0) segs.splice(i, 1); }
    return o.before + segs.map(s => s.raw).join(',') + o.after;
  }
  const reindent = (lines, from, to) => from === to ? lines : lines.map(l => (isBlank(l) ? l : ' '.repeat(to) + l.slice(Math.min(from, indentOf(l)))));
  const keyOf = text => { const m = /^self\.(\w+)\s*(?::[^=]+)?=(?!=)/.exec(text); return m ? m[1] : null; };
  const jn = (lines, a, b) => lines.slice(a, b).join('\n');

  // Rewrite one method of the original class so it follows the generated base -> cur change.
  function patchMethod(oL, oIt, bL, bIt, cL, cIt, ci) {
    const O = methodParts(oL, oIt, ci.o), B = methodParts(bL, bIt, ci.b), C = methodParts(cL, cIt, ci.c);
    const head = patchParams(jn(oL, O.head[0], O.head[1]), jn(bL, B.head[0], B.head[1]), jn(cL, C.head[0], C.head[1])).split('\n');
    const lead = oL.slice(oIt.a, oIt.p);
    const name = (/^(?:async\s+)?def\s+(\w+)/.exec(oIt.text) || [])[1];
    if (name !== '__init__') {
      // Replace the body with the generated one, keeping the original docstring.
      const doc = O.docstring ? oL.slice(O.docstring.a, O.body.length > 1 ? O.body[1].a : O.docstring.b) : [];
      const from = C.docstring ? C.docstring.b : C.bodyFrom;
      const gen = reindent(cL.slice(from, C.body.length ? C.body[C.body.length - 1].b : from), C.indent, O.indent);
      while (gen.length && isBlank(gen[0])) gen.shift();
      return lead.concat(head, doc, gen);
    }
    // __init__: change only the `self.x = ...` statements that differ.
    const key = (L, M) => new Map(M.body.map(it => [keyOf(it.text), it]).filter(([k]) => k));
    const ok = key(oL, O), bk = key(bL, B), ck = key(cL, C);
    const stmt = (L, it, M) => reindent(L.slice(it.p, it.b), M.indent, O.indent);
    const replace = new Map(), drop = new Set(), after = new Map();
    for (const [k, it] of ck) {
      const bi = bk.get(k), oi = ok.get(k);
      if (bi && jn(bL, bi.p, bi.b) === jn(cL, it.p, it.b)) continue;
      if (oi) replace.set(oi, oL.slice(oi.a, oi.p).concat(stmt(cL, it, C)));
      else if (!bi) {
        let prev = null; for (const [pk] of ck) { if (pk === k) break; if (ok.has(pk)) prev = ok.get(pk); }
        const sup = O.body.find(x => /^super\(/.test(x.text)) || O.docstring || O.body[0];
        const anchor = prev || sup;
        if (anchor) after.set(anchor, (after.get(anchor) || []).concat(stmt(cL, it, C)));
        else after.set('start', (after.get('start') || []).concat(stmt(cL, it, C)));
      }
    }
    for (const [k, it] of bk) if (!ck.has(k) && ok.has(k)) drop.add(ok.get(k));
    const out = lead.concat(head);
    const end = O.body.length ? O.body[O.body.length - 1].b : O.bodyFrom;
    const byStart = new Map(O.body.map(it => [it.a, it]));
    if (after.has('start')) out.push(...after.get('start'));
    for (let i = O.bodyFrom; i < end; i++) {
      const it = byStart.get(i);
      if (it) {
        if (drop.has(it)) { i = it.b - 1; if (i + 1 < end && isBlank(oL[i + 1])) i++; continue; }
        if (replace.has(it)) { out.push(...replace.get(it)); i = it.b - 1; }
        else { out.push(...oL.slice(it.a, it.b)); i = it.b - 1; }
        if (after.has(it)) out.push(...after.get(it));
        continue;
      }
      out.push(oL[i]);
    }
    return out;
  }
  // Patch one class: keep every method the diagram change did not touch.
  function patchClass(oL, bL, cL) {
    const O = classParts(oL), B = classParts(bL), C = classParts(cL);
    const ci = { o: O.indent, b: B.indent, c: C.indent };
    const norm = (L, it) => jn(L, it.p, it.b).replace(/[ \t]+$/gm, '');
    const bm = new Map(B.methods.map(m => [m.name, m])), om = new Map(O.methods.map(m => [m.name, m])), cm = new Map(C.methods.map(m => [m.name, m]));
    const edits = [];
    for (const [name, c] of cm) {
      const b = bm.get(name), o = om.get(name);
      if (b && norm(bL, b) === norm(cL, c)) continue;
      if (o && b) edits.push({ a: o.a, b: o.b, lines: patchMethod(oL, o, bL, b, cL, c, ci) });
      else if (o) edits.push({ a: o.a, b: o.b, lines: reindent(cL.slice(c.a, c.b), C.indent, O.indent) });
      else {
        const last = O.methods.length ? O.methods[O.methods.length - 1].b : oL.length;
        edits.push({ a: last, b: last, lines: [''].concat(reindent(cL.slice(c.a, c.b), C.indent, O.indent)) });
      }
    }
    // a method the diagram removed is deleted from the class
    for (const name of bm.keys()) if (!cm.has(name) && om.has(name)) { const o = om.get(name); edits.push({ a: o.a, b: o.b, lines: [], blank: true }); }
    edits.sort((x, y) => y.a - x.a);
    const out = oL.slice();
    for (const e of edits) out.splice(e.a, e.b - e.a, ...e.lines);
    return out;
  }
  function ensureImports(text, code) {
    const need = IMPORTS.filter(([, have, uses]) => uses.test(code) && !have.test(text)).map(i => i[0]);
    if (!need.length) return text;
    const lines = text.split('\n'), m = /^(\w+)\s*=\s*marimo\.App\(/m.exec(text);
    if (m) {
      const at = lines.findIndex(l => /^with\s+\w+\.setup\b.*:\s*$/.test(l));
      const add = need.map(l => '    ' + l);
      if (at >= 0) {
        let end = at + 1, last = at;
        while (end < lines.length && (isBlank(lines[end]) || /^\s/.test(lines[end]))) { if (/^\s+(import|from)\s/.test(lines[end])) last = end; end++; }
        lines.splice(last + 1, 0, ...add);
      } else {
        const appAt = lines.findIndex(l => /^\w+\s*=\s*marimo\.App\(/.test(l));
        lines.splice(appAt + 1, 0, '', `with ${m[1]}.setup:`, ...add);
      }
      return lines.join('\n');
    }
    let at = -1;
    for (let i = 0; i < lines.length; i++) { if (/^(import|from)\s/.test(lines[i])) at = i; else if (/^(class|def|if|@)/.test(lines[i])) break; }
    lines.splice(at + 1, 0, ...need);
    return lines.join('\n');
  }
  // Apply the diagram edit `graph` to `text`, the current file contents. Returns the new text.
  // `o.root` picks the model class, `o.gen` holds the code-style options (as genOpts() in src.html).
  function syncSource(text, graph, o = {}) {
    const src = text.replace(/\r\n/g, '\n');
    const res = importPython(src, { root: o.root });
    // Compare in the file's own style: a literal file has no size symbols to patch, so comparing symbolic
    // code would drop a size edit made through a symbol.
    const opts = Object.assign({}, o.gen, { shapeComments: false, literalDims: res.literalDims != null ? res.literalDims : false });
    const baseGen = generate(res.graph, Object.assign({}, opts, { className: res.graph.name })).code;
    const curGen = generate(graph, Object.assign({}, opts, { className: graph.name })).code;
    if (baseGen === curGen) return src;
    const lines = src.split('\n'), bl = baseGen.split('\n'), cl = curGen.split('\n');
    const marimo = /^(\w+)\s*=\s*marimo\.App\(/m.exec(src), app = marimo && marimo[1];
    const oBlocks = classBlocks(lines), bBlocks = classBlocks(bl), cBlocks = classBlocks(cl);
    const find = (blocks, n) => blocks.find(x => x.name === n);
    const edits = [];
    const deco = (cls, extra) => (app ? [`@${app}.${extra}`] : []).concat(cls);
    for (const c of cBlocks) {
      // renaming the model in the diagram renames its root class
      const was = c.name === graph.name ? res.graph.name : c.name;
      const b = find(bBlocks, was), ob = find(oBlocks, was), clsLines = cl.slice(c.h, c.b);
      if (b && jn(bl, b.h, b.b) === jn(cl, c.h, c.b)) continue;
      if (ob && b) {
        const patched = patchClass(lines.slice(ob.h, ob.b), bl.slice(b.h, b.b), clsLines);
        patched[0] = patched[0].replace(/^class\s+\w+/, 'class ' + c.name);
        edits.push({ a: ob.h, b: ob.b, lines: patched });
      }
      else if (!ob) {
        const used = oBlocks.filter(x => (res.usedClasses || []).includes(x.name));
        const before = used.find(x => cBlocks.findIndex(z => z.name === x.name) > cBlocks.indexOf(c));
        const at = before ? before.a : used.length ? Math.max(...used.map(x => x.b)) : lines.length;
        edits.push({ a: at, b: at, lines: (before ? [] : ['', '']).concat(deco(clsLines, 'class_definition'), before ? ['', ''] : []) });
      }
    }
    const renamedFrom = graph.name !== res.graph.name && find(cBlocks, graph.name) ? res.graph.name : null;
    // a class the diagram no longer has is deleted, unless it was renamed
    for (const b of bBlocks) if (!find(cBlocks, b.name) && b.name !== renamedFrom) {
      const ob = find(oBlocks, b.name);
      // the blank lines after it go too, so the gap before it stays as it was
      if (ob) { let e = ob.b; while (e < lines.length - 1 && isBlank(lines[e])) e++; edits.push({ a: ob.a, b: e, lines: [] }); }
    }
    // helper functions the generated code needs
    const helperDefs = cl.slice(0, Math.max(0, cl.findIndex(l => /^class /.test(l)))).join('\n').split(/\n(?=def )/).map(s => s.trim()).filter(s => /^def (\w+)/.test(s));
    const missing = helperDefs.filter(s => !new RegExp('^def ' + /^def (\w+)/.exec(s)[1] + '\\(', 'm').test(src));
    if (missing.length) {
      const first = oBlocks.length ? oBlocks[0].a : lines.length;
      edits.push({ a: first, b: first, lines: missing.flatMap(s => deco(s.split('\n'), 'function').concat(['', ''])) });
    }
    edits.sort((x, y) => y.a - x.a || y.b - x.b);
    const out = lines.slice();
    for (const e of edits) out.splice(e.a, e.b - e.a, ...e.lines);
    return ensureImports(out.join('\n'), edits.flatMap(e => e.lines).join('\n'));
  }

  return { HELPER_OPS, helperDoc, srcKey, splitKey, blockOutputs, blockClassCode, LOSSES, lossParts, lossPlan, weightShape, tieOwner, blockParams, blockInputs, expandBlocks, instantiateBlock, presetResidualMLPBlocks, NORMS, linkSizes, syncWidths, mergeIntoSource, syncSource, isMarimo, uid, BATCH, DEFAULTS, ACTS, INITS, initCode, makeNode, inCount, isCircle, index, topo, evalDim, parseShape, inferShapes, paramCount, propagateWidth, generate, importPython, importFx, autoLayout, nodeSize, nodeLabel, presetResidualMLP, parse, identsIn };
})();
if (typeof module !== 'undefined') module.exports = NN;
