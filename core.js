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
    // bias_init 'dt': Mamba's Δ bias, softplus(bias) log-uniform in [0.001, 0.1]
    linear: { in: 'f', out: 'f', bias: false, init: 'default', init_arg: '', zero_bias: true, bias_init: '', tie: '', name: '', title: '', var: '' },
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
    transpose: { title: '', var: '' },
    // a rearrangement of the axes, einops-style: 'e t (h d) -> e h t d' with sizes 'h=n_head' (see reshapePlan)
    reshape: { pattern: '... a b -> ... b a', sizes: '', note: '', title: '', var: '' },                // swap the last two dims (Kᵀ)
    scale: { title: '', var: '' },                    // × 1/√(last dim)
    mask: { mode: 'causal', title: '', var: '' },     // causal mask: future positions → −inf; 'given': an allowed mask comes in on input 2
    rope: { base: '10000', mode: 'seq', sections: '16, 24, 24', title: '', var: '' }, // rotary position embedding on (…, T, D); 'axial'/'mrope': positions on input 2
    // images (e, C, H, W): a 2-D convolution, pooling, and nearest/bilinear upsampling
    // transposed: nn.ConvTranspose2d, which enlarges the map: (H − 1)·stride − 2·pad + k (U-Net's 2×2, stride 2 doubles it)
    conv: { in: 'c', out: 'c', k: '3', stride: '1', pad: '1', bias: false, transposed: false, name: '', title: '', var: '' },
    pool: { kind: 'max', k: '2', stride: '', pad: '0', size: '1', title: '', var: '' }, // max / avg: k, stride (default k), pad; adaptive: output size
    upsample: { scale: '2', mode: 'nearest', title: '', var: '' },
    // one time step later: out[:, t] = in[:, t − 1], zeros at t = 0 (e × t × …). A loop through a Delay is a recurrence,
    // computed step by step; name is the state's name in the code (h, c)
    delay: { name: 'h', title: '', var: '' },
    // learned vectors put in front of the sequence (ViT's class token): (e, t, d) → (e, n + t, d), starting at zeros
    token: { n: '1', dim: 'd', name: '', title: '', var: '' },
    // one position of a sequence: x[:, k], (e, t, …) → (e, …)
    select: { index: '0', title: '', var: '' },
    // a learned tensor read as it is (Mamba's A_log and D): zeros, ones, or 's4d' (log 1 … N along the last size,
    // the same for every row: Mamba's S4D-real init of A)
    // also 'normal' (std init_arg) and 'sincos2d' (fixed 2-D sine/cosine positions of a square patch grid, I-JEPA)
    param: { shape: 'd', init: 'zeros', init_arg: '', name: '', title: '', var: '' },
    // a convolution along time on (e, t, ch), one filter per channel, that only sees the current and earlier steps
    // (Mamba's conv1d): nn.Conv1d(ch, ch, k, groups=ch, padding=k − 1), cut back to t steps
    conv1d: { ch: 'd', k: '4', bias: true, name: '', title: '', var: '' },
    // standard Gaussian noise shaped like its input (a VAE's ε in z = μ + ε·σ): torch.randn_like(x)
    noise: { title: '', var: '' },
    // loss: port 0 = a model output (raw, before any final activation), port 1 = the target
    loss: { fn: 'cross_entropy', mode: 'given', label_smoothing: '', ignore_index: '', delta: '1.0', weight: '1', mask_prob: '0.15', mask_id: '', title: '' },
  };
  const ACTS = ['sigmoid', 'silu', 'relu', 'leaky_relu', 'gelu', 'gelu_tanh', 'tanh', 'softmax', 'softplus'];
  // Loss functions. act = what predict() applies at inference (the loss works on the raw output).
  const LOSSES = {
    cross_entropy: { label: 'Cross-entropy', torch: 'F.cross_entropy', act: 'softmax', note: 'Classes or tokens; takes logits' },
    bce: { label: 'Binary cross-entropy', torch: 'F.binary_cross_entropy_with_logits', act: 'sigmoid', note: 'Yes/no per output; takes logits' },
    mse: { label: 'Mean squared error', torch: 'F.mse_loss', act: null, note: 'Regression' },
    l1: { label: 'L1 (mean absolute error)', torch: 'F.l1_loss', act: null, note: 'Regression, robust to outliers' },
    huber: { label: 'Huber', torch: 'F.huber_loss', act: null, note: 'MSE near zero, L1 far away' },
    // port 0: the mean, port 1: the log-variance, both model outputs (no target)
    kl: { label: 'KL to N(0, 1)', torch: 'kl_normal', act: null, note: 'VAE: a Gaussian\'s mean (input 1) and log-variance (input 2)' },
  };
  const KL_HELPER = `def kl_normal(mu, logvar):\n    \"\"\"KL divergence from N(mu, exp(logvar)) to N(0, 1): summed over the latent features, averaged over the batch.\"\"\"\n    return -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))`;
  // Normalization layers. channelFirst: PyTorch expects (N, C, L), so (e, t, d) inputs are transposed around the call.
  const NORMS = {
    LayerNorm: { label: 'LayerNorm', note: 'Per example, over the features', short: 'LayerNorm' },
    RMSNorm: { label: 'RMSNorm', note: 'LayerNorm without mean-centering or bias', short: 'RMSNorm' },
    BatchNorm1d: { label: 'BatchNorm', note: 'Per feature, over the batch (running stats)', short: 'BatchNorm', channelFirst: true },
    GroupNorm: { label: 'GroupNorm', note: 'Per example, over groups of features', short: 'GroupNorm', channelFirst: true },
    InstanceNorm1d: { label: 'InstanceNorm', note: 'Per example and feature, over the sequence', short: 'InstanceNorm', channelFirst: true },
    // images: per channel (dim 1 of e × C × H × W), over the batch and the pixels
    BatchNorm2d: { label: 'BatchNorm2d', note: 'Per channel of an image, over the batch and pixels (running stats)', short: 'BatchNorm2d', image: true },
  };
  // the axis a norm works on: the channels (dim 1) for image norms, otherwise the last
  const normAxis = (p, x) => (NORMS[p.kind] && NORMS[p.kind].image ? 1 : x.length - 1);
  const normCtor = (p, d) => {
    const eps = p.eps !== '' && p.eps != null ? `, eps=${p.eps}` : '';
    switch (p.kind) {
      case 'LayerNorm': return `nn.LayerNorm(${d}${eps}${p.affine === false ? ', elementwise_affine=False' : ''})`;
      case 'BatchNorm1d': return `nn.BatchNorm1d(${d}${eps})`;
      case 'BatchNorm2d': return `nn.BatchNorm2d(${d}${eps})`;
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
      case 'input': case 'param': return 0;
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
  // A Delay's input is the previous step's value, so it doesn't have to come first: loops through a Delay are fine
  function topo(g, ix = index(g)) {
    const indeg = new Map(g.nodes.map(n => [n.id, 0]));
    for (const n of g.nodes) if (n.type !== 'delay') for (const s of ix.ins.get(n.id)) if (s) indeg.set(n.id, indeg.get(n.id) + 1);
    const q = g.nodes.filter(n => indeg.get(n.id) === 0).map(n => n.id);
    const order = [];
    while (q.length) {
      const id = q.shift(); order.push(id);
      for (const o of ix.outs.get(id)) { const c = ix.byId.get(o.to); if (c && c.type === 'delay') continue; indeg.set(o.to, indeg.get(o.to) - 1); if (indeg.get(o.to) === 0) q.push(o.to); }
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
  // With Delays, a loop's sizes depend on themselves: each Delay takes the shape that reached it last time round,
  // until nothing changes (a couple of passes)
  function inferShapes(g, depth = 0) {
    if (!g.nodes.some(n => n.type === 'delay')) return inferCore(g, depth, null);
    let pre = new Map(), r;
    for (let k = 0; k < 8; k++) {
      r = inferCore(g, depth, pre);
      const ix = index(g), nxt = new Map(); let same = true;
      for (const n of g.nodes) if (n.type === 'delay') {
        const s = ix.inKeys.get(n.id)[0], sh = (s && r.shapes.get(s)) || null;
        nxt.set(n.id, sh); if (JSON.stringify(sh) !== JSON.stringify(pre.get(n.id) || null)) same = false;
      }
      if (same) break; pre = nxt;
    }
    return r;
  }
  function inferCore(g, depth, pre) {
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
      if (n.type === 'delay') {
        if (!ix.ins.get(id)[0]) err(id, 'Unconnected input');
        const sh = (pre && pre.get(id)) || null;
        if (sh && sh.length < 2) conflict(id, { kind: 'rank', a: sh.join('×'), b: 'e × t × …', ports: [0], cap: 'needs a sequence', msg: 'A Delay shifts along time, so its input needs a shape like e × t × d' });
        shapes.set(id, sh); continue;
      }
      const inS = ix.inKeys.get(id).map(s => (s ? shapes.get(s) : undefined));
      if (n.type !== 'input' && inS.some(s => s === undefined)) {
        if (ix.ins.get(id).some(s => !s)) err(id, 'Unconnected input');
        shapes.set(id, null); continue;
      }
      // around a loop, the first pass has no shape yet for what comes back through a Delay: an elementwise op
      // takes its other input's (the next pass checks them properly)
      if (pre && n.type === 'binop' && inS.some(s => s === null) && inS.some(s => s)) { const k = inS.find(s => s); for (let i = 0; i < inS.length; i++) if (!inS[i]) inS[i] = k; }
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
          const x = inS[0], C = x[normAxis(p, x)];
          if (NORMS[p.kind] && NORMS[p.kind].image && x.length !== 4) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × C × H × W', ports: [0], cap: 'needs an image', msg: `${NORMS[p.kind].label} works on images shaped e × C × H × W` }); out = x; break; }
          if (dimEq(C, p.dim, sym) === false) err(id, `Normalizes ${p.dim}, got ${C}`);
          if (p.kind === 'GroupNorm') {
            const c = evalDim(C, sym), gN = +p.groups || 8;
            if (c != null && c % gN !== 0) conflict(id, { kind: 'groups', a: C, b: String(gN), ports: [0], cap: `${C} (${c}) is not divisible by ${gN} groups`, msg: `GroupNorm needs the width to divide into ${gN} groups, but ${C} = ${c}` });
          }
          if (p.kind === 'InstanceNorm1d' && x.length < 3) conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × t × d', ports: [0], cap: 'needs a sequence dimension', msg: 'InstanceNorm normalizes over a sequence, so its input needs a shape like e × t × d' });
          if (NORMS[p.kind] && NORMS[p.kind].channelFirst && x.length > 3) conflict(id, { kind: 'rank', a: x.join('×'), b: 'rank ≤ 3', ports: [0], cap: 'input has more than 3 dims', msg: `${NORMS[p.kind].label} here supports inputs of shape e × d or e × t × d` });
          out = x; break;
        }
        case 'act': case 'dropout': case 'scale': case 'noise': out = inS[0]; break;
        case 'token': {
          const x = inS[0];
          if (x.length < 3) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × t × d', ports: [0], cap: 'needs a sequence', msg: 'A class token goes in front of a sequence shaped e × t × d' }); out = x; break; }
          if (dimEq(last(x), p.dim, sym) === false) err(id, `Its tokens have ${p.dim} features, the sequence has ${last(x)}`);
          const T = x[1], k = String(p.n || '1');
          out = [x[0], /^\d+$/.test(String(T)) && /^\d+$/.test(k) ? String(+T + +k) : `${par0(T)} + ${k}`, ...x.slice(2)]; break;
        }
        case 'param': out = parseShape(p.shape); break;
        case 'conv1d': {
          const x = inS[0];
          if (x.length < 3) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × t × ch', ports: [0], cap: 'needs a sequence', msg: 'A causal convolution runs along a sequence shaped e × t × ch' }); out = x; break; }
          if (dimEq(last(x), p.ch, sym) === false) err(id, `It has ${p.ch} channels, the sequence has ${last(x)}`);
          out = x; break;
        }
        case 'select': {
          const x = inS[0];
          if (x.length < 2) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × t × …', ports: [0], cap: 'needs a sequence', msg: 'Picking a position needs a sequence shaped e × t × …' }); out = x; break; }
          out = [x[0], ...x.slice(2)]; break;
        }
        case 'conv': {
          const x = inS[0];
          if (x.length !== 4) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × C × H × W', ports: [0], cap: 'needs an image', msg: 'A 2-D convolution works on images shaped e × C × H × W' }); out = x; break; }
          if (dimEq(x[1], p.in, sym) === false) err(id, `Expects ${p.in} channels, got ${x[1]}`);
          const sz = p.transposed ? d => {
            const H = evalDim(d, sym), k = evalDim(p.k, sym), s2 = evalDim(p.stride, sym), q = evalDim(p.pad, sym);
            if (/^\d+$/.test(String(d)) && [H, k, s2, q].every(v => v != null)) return String((H - 1) * s2 - 2 * q + k);
            return String(p.k) === String(p.stride) && String(p.pad) === '0' ? `${par0(d)}*${par0(p.stride)}` : `(${d}-1)*${par0(p.stride)}-2*${par0(p.pad)}+${par0(p.k)}`;
          } : d => spatial(d, p.k, p.stride, p.pad);
          out = [x[0], String(p.out), sz(x[2]), sz(x[3])]; break;
        }
        case 'pool': {
          const x = inS[0];
          if (x.length !== 4) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × C × H × W', ports: [0], cap: 'needs an image', msg: 'Pooling works on images shaped e × C × H × W' }); out = x; break; }
          if (p.kind === 'adaptive') { const z = String(p.size || '1'); out = [x[0], x[1], z, z]; break; }
          const st = p.stride !== '' && p.stride != null ? p.stride : p.k;
          out = [x[0], x[1], spatial(x[2], p.k, st, p.pad), spatial(x[3], p.k, st, p.pad)]; break;
        }
        case 'upsample': {
          const x = inS[0];
          if (x.length !== 4) { conflict(id, { kind: 'rank', a: x.join('×'), b: 'e × C × H × W', ports: [0], cap: 'needs an image', msg: 'Upsampling works on images shaped e × C × H × W' }); out = x; break; }
          const up = d => { const v = evalDim(d, sym), f = evalDim(p.scale, sym); return v != null && f != null && /^\d+$/.test(String(d)) ? String(v * f) : `${par0(d)}*${p.scale}`; };
          out = [x[0], x[1], up(x[2]), up(x[3])]; break;
        }
        case 'loss': {
          const [P, Y] = inS, fn = p.fn || 'cross_entropy', mode = fn === 'cross_entropy' ? p.mode || 'given' : 'given';
          const same = (a, b) => a.length === b.length && a.every((d, i) => dimEq(d, b[i], sym) !== false);
          const txt = x => x.map(show).join(' × ');
          if (fn === 'cross_entropy') {
            const want = P.length === 4 ? [P[0], P[2], P[3]] : P.slice(0, -1); // an image's classes are its channels
            if (mode === 'next' && P.length < 3) conflict(id, { kind: 'loss', a: txt(P), b: 'e × t × classes', ports: [0], cap: 'needs a sequence', msg: 'Next-token loss needs sequence logits (e × t × vocab)' });
            else if (!same(want, Y)) conflict(id, { kind: 'loss', a: txt(Y), b: txt(want), ports: [1], cap: `target should be ${txt(want)}`, msg: `Cross-entropy takes class indices shaped like the logits without the class dimension: the target is ${txt(Y)}, it should be ${txt(want)}` });
          } else if (fn === 'kl' && !same(P, Y)) conflict(id, { kind: 'loss', a: txt(Y), b: txt(P), ports: [1], cap: `log-variance should be ${txt(P)}`, msg: `The KL loss takes a mean and a log-variance of the same shape: ${txt(P)} and ${txt(Y)}` });
          else if (fn !== 'kl' && !same(P, Y)) conflict(id, { kind: 'loss', a: txt(Y), b: txt(P), ports: [1], cap: `target should be ${txt(P)}`, msg: `${(LOSSES[fn] || {}).label || 'This loss'} compares two tensors of the same shape: the target is ${txt(Y)}, the output is ${txt(P)}` });
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
          out = p.positions ? [x.length >= 2 ? x[1] : x[0], String(p.dim)] : x.concat(String(p.dim));
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
        case 'reshape': {
          const x = inS[0];
          try {
            const r = reshapeShape(p, x); out = r.out;
            // a split must divide: 512 into h = 8 parts of 64
            if (r.LL) r.LL.forEach((it, i) => { if (!it.g) return; const D = evalDim(x[i], sym), ks = it.g.filter(n => r.sizes[n] != null).map(n => evalDim(r.sizes[n], sym)); const k = ks.reduce((t, v) => (t == null || v == null ? null : t * v), 1);
              if (D != null && k != null && (k <= 0 || D % k !== 0)) conflict(id, { kind: 'reshape', a: x[i], b: it.g.filter(n => r.sizes[n] != null).map(n => r.sizes[n]).join('·'), ports: [0], msg: `Axis ${i + 1} (${x[i]}) doesn't split into (${it.g.join(' ')}) with ${it.g.filter(n => r.sizes[n] != null).map(n => n + ' = ' + r.sizes[n]).join(', ')}` }); });
          } catch (e2) { err(id, e2.message); out = x; }
          break;
        }
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
          if (/(^|\.)(sum|mean|amax|prod)$/.test(fn) && /^-?\d+$/.test(ex) && inS.length === 1) { const k = ax(+ex); out = x0.slice(0, k).concat(x0.slice(k + 1)); break; }
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
            if (mm) { const s0 = ax(+mm[1]), s1 = mm[2] != null ? ax(+mm[2]) : x0.length - 1, part = x0.slice(s0, s1 + 1); out = x0.slice(0, s0).concat(part.every(d => /^\d+$/.test(String(d))) ? String(part.reduce((t, d) => t * +d, 1)) : part.map(par).join('*'), x0.slice(s1 + 1)); break; }
          }
          if (/(^|\.)transpose$/.test(fn) && inS.length === 1) { const mm = /^(-?\d+)\s*,\s*(-?\d+)$/.exec(ex); if (mm) { const a = ax(+mm[1]), b = ax(+mm[2]); out = x0.slice(); [out[a], out[b]] = [out[b], out[a]]; break; } }
          if (/(^|\.)permute$/.test(fn) && inS.length === 1) { const pm = ex.split(',').map(v => +v.trim()); if (pm.length === x0.length && pm.every(v => Number.isInteger(v))) { out = pm.map(i => x0[ax(i)]); break; } }
          if (/(^|\.)(matmul|mm|bmm)$/.test(fn) && inS.length === 2) { out = matmul(id, inS[0], inS[1]); break; }
          if (/(^|\.)(cat|concat|concatenate)$/.test(fn) && inS.length > 1) {
            const ax = m ? +m[1] : 0, r0 = inS[0], k = ax < 0 ? r0.length + ax : ax;
            for (let j = 1; j < inS.length && !conflicts.has(id); j++) {
              const r1 = inS[j];
              if (r1.length !== r0.length) { conflict(id, { kind: 'cat', a: r0.join('×'), b: r1.join('×'), ports: [0, j], msg: `Concatenated tensors have different ranks: ${r0.join(' × ')} vs ${r1.join(' × ')}` }); break; }
              r0.forEach((d, i) => { if (i !== k && !conflicts.has(id) && dimEq(d, r1[i], sym) === false) conflict(id, { kind: 'cat', a: d, b: r1[i], ports: [0, j], msg: `Sizes disagree outside the concat axis: ${show(d)} vs ${show(r1[i])}` }); });
            }
            // literal sizes add up to a number (256 + 512 → 768); named ones stay a sum
            out = r0.map((d, i) => (i !== k ? d : inS.every(s => /^\d+$/.test(String(s[i]))) ? String(inS.reduce((t, s) => t + +s[i], 0)) : inS.map(s => s[i]).join(' + ')));
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
    // output size of a convolution / pooling window along one axis: ⌊(H + 2·pad − k) / stride⌋ + 1
    function spatial(H, k, st, pd) {
      const h = evalDim(H, sym), kk = evalDim(k, sym), s = evalDim(st, sym), q = evalDim(pd, sym);
      if (h != null && /^\d+$/.test(String(H)) && kk != null && s != null && q != null && s > 0) return String(Math.floor((h + 2 * q - kk) / s) + 1);
      // named sizes: 'same' padding (pad = (k − 1) / 2) keeps the size at stride 1 and divides it by the stride
      const same = kk != null && q != null && kk === 2 * q + 1;
      if (same || (String(k) === String(st) && q === 0)) return s === 1 ? String(H) : `${par0(H)}/${par0(st)}`;
      return `(${H}+2*${par0(pd)}-${par0(k)})//${par0(st)}+1`;
    }
  }
  function paramCount(n, symbols) {
    const p = n.params, d = x => evalDim(x, symbols);
    if (n.type === 'linear') { const i = d(p.in), o = d(p.out); return i == null || o == null ? null : (p.tie ? 0 : i * o) + (p.bias ? o : 0); }
    if (n.type === 'embed') { const a = d(p.num), b = d(p.dim); return a == null || b == null ? null : p.tie ? 0 : a * b; }
    if (n.type === 'norm') { const k = d(p.dim); if (p.affine === false) return 0; return k == null ? null : ({ LayerNorm: 2 * k, BatchNorm1d: 2 * k, BatchNorm2d: 2 * k, GroupNorm: 2 * k, InstanceNorm1d: 0 }[p.kind] ?? k); }
    if (n.type === 'token') { const a = d(p.n), b = d(p.dim); return a == null || b == null ? null : a * b; }
    if (n.type === 'param') { const v = parseShape(p.shape).map(d); return v.some(x => x == null) ? null : v.reduce((t, x) => t * x, 1); }
    if (n.type === 'conv1d') { const c = d(p.ch), k = d(p.k); return c == null || k == null ? null : c * k + (p.bias ? c : 0); }
    if (n.type === 'conv') { const i = d(p.in), o = d(p.out), k = d(p.k); return i == null || o == null || k == null ? null : i * o * k * k + (p.bias ? o : 0); }
    return 0;
  }

  // A Linear's input width and a Norm's width always follow the tensor flowing in,
  // so sizes are only ever set where they are created (Linear outputs, inputs).
  function syncWidths(g) {
    const { shapes } = inferShapes(g), ix = index(g); let changed = 0;
    for (const n of g.nodes) {
      if (n.type !== 'linear' && n.type !== 'norm' && n.type !== 'conv' && n.type !== 'token' && n.type !== 'conv1d') continue;
      const src = ix.inKeys.get(n.id)[0], sh = src && shapes.get(src);
      if (!sh || !sh.length) continue;
      // a convolution and an image norm take the channels (dim 1); the others the last size
      const ax = n.type === 'conv' ? 1 : n.type === 'norm' ? normAxis(n.params, sh) : sh.length - 1;
      if (sh[ax] == null) continue;
      const d = String(sh[ax]), f = n.type === 'norm' || n.type === 'token' ? 'dim' : n.type === 'conv1d' ? 'ch' : 'in', cur = String(n.params[f]);
      // a layer keeps its own plain size when what arrives is a formula worth the same (heads × head width = d)
      if (cur !== d && /^[A-Za-z_]\w*$|^\d+$/.test(cur) && !/^[A-Za-z_]\w*$|^\d+$/.test(d) && dimEq(cur, d, g.symbols) === true) continue;
      // a convolution or image norm keeps its own named channel count while what arrives is worth the same
      // (a block's c_in, fed 64 channels from the layer before)
      if ((n.type === 'conv' || (n.type === 'norm' && (NORMS[n.params.kind] || {}).image)) && cur !== d && /^[A-Za-z_]\w*$/.test(cur) && dimEq(cur, d, g.symbols) === true) continue;
      if (cur !== d) { n.params[f] = d; changed++; }
    }
    return changed;
  }

  // ── Reshape: one node for every rearrangement of a tensor's axes, written einops-style ──
  // params.pattern 'e t (h d) -> e h t d', params.sizes 'h=n_head'. The names are local labels for the axes: '(h d)' is
  // one axis made of two, '1' an axis of size 1, '...' any axes not named. Only the arrangement changes, never a value.
  // In code it is ordinary PyTorch: unflatten, squeeze, transpose / permute, flatten, unsqueeze, in that order.
  function parseSide(s) {
    const out = []; let grp = null;
    const toks = String(s).replace(/…/g, '...').replace(/\(/g, ' ( ').replace(/\)/g, ' ) ').trim().split(/\s+/).filter(Boolean);
    for (const t of toks) {
      if (t === '(') { if (grp) throw Error('Groups ( ) can\'t nest in a reshape pattern'); grp = []; continue; }
      if (t === ')') { if (!grp) throw Error('An unmatched ) in the reshape pattern'); if (!grp.length) throw Error('An empty ( ) in the reshape pattern'); out.push({ g: grp }); grp = null; continue; }
      if (!/^([A-Za-z_]\w*|1|\.\.\.)$/.test(t)) throw Error(`"${t}" is not an axis name (use letters, 1 or ...)`);
      if (grp) { if (t === '...' || t === '1') throw Error(`"${t}" can't go inside ( )`); grp.push(t); }
      else out.push(t === '...' ? { ell: true } : t === '1' ? { one: true } : { n: t });
    }
    if (grp) throw Error('An unmatched ( in the reshape pattern');
    return out;
  }
  function parseSizes(s) {
    const sizes = {};
    String(s || '').split(/[,;]/).map(x => x.trim()).filter(Boolean).forEach(x => { const m = /^([A-Za-z_]\w*)\s*=\s*(.+)$/.exec(x); if (!m) throw Error(`"${x}": write sizes like h = n_head`); sizes[m[1]] = m[2].trim(); });
    return sizes;
  }
  // the pattern for an input of rank rIn: both sides with '...' expanded
  function reshapePlan(p, rIn) {
    const parts = String(p.pattern || '').split(/->|→/);
    if (parts.length !== 2) throw Error('A reshape pattern reads "before -> after", e.g. e t (h d) -> e h t d');
    const L = parseSide(parts[0]), R = parseSide(parts[1]), sizes = parseSizes(p.sizes);
    const ellL = L.filter(x => x.ell).length, ellR = R.filter(x => x.ell).length;
    if (ellL > 1 || ellR > 1 || ellL !== ellR) throw Error('Use ... once on each side, or not at all');
    const nL = L.length - ellL;
    if (!ellL && nL !== rIn) throw Error(`The left side names ${nL} axes; the tensor has ${rIn}`);
    if (ellL && nL > rIn) throw Error(`The left side names ${nL} axes; the tensor has only ${rIn}`);
    const rest = Array.from({ length: rIn - nL }, (_, i) => '…' + i), ex = side => side.flatMap(x => (x.ell ? rest.map(n => ({ n })) : [x]));
    const LL = ex(L), RR = ex(R);
    const elems = side => side.flatMap(x => (x.g ? x.g : x.n ? [x.n] : []));
    const eL = elems(LL), eR = elems(RR), dup = a => a.find((x, i) => a.indexOf(x) !== i);
    if (dup(eL)) throw Error(`"${dup(eL)}" appears twice on the left`);
    if (dup(eR)) throw Error(`"${dup(eR)}" appears twice on the right`);
    const miss = eL.find(x => !eR.includes(x)) || eR.find(x => !eL.includes(x));
    if (miss) throw Error(`"${miss}" must appear on both sides (a reshape keeps every value)`);
    for (const k of Object.keys(sizes)) if (!eL.includes(k)) throw Error(`The size ${k} = ${sizes[k]} names no axis`);
    return { LL, RR, sizes };
  }
  // PyTorch steps, each with the rank it applies to
  function reshapeSteps(p, rIn) {
    if (Array.isArray(p._steps)) return p._steps;
    const { LL, RR, sizes } = reshapePlan(p, rIn), steps = [];
    let cur = LL.map(x => (x.g ? { names: x.g, split: true } : x.one ? { one: true } : { names: [x.n] }));
    for (let i = 0; i < cur.length; i++) if (cur[i].split) {
      const nm = cur[i].names, unknown = nm.filter(n => sizes[n] == null);
      if (unknown.length > 1) throw Error(`Give the size of ${unknown.slice(0, -1).join(', ')} (e.g. ${unknown[0]} = 8): only one part of (${nm.join(' ')}) can be worked out`);
      steps.push({ op: 'unflatten', dim: i, sizes: nm.map(n => (sizes[n] != null ? sizes[n] : -1)), rank: cur.length });
      cur.splice(i, 1, ...nm.map(n => ({ names: [n] }))); i += nm.length - 1;
    }
    for (let i = cur.length - 1; i >= 0; i--) if (cur[i].one) { steps.push({ op: 'squeeze', dim: i, rank: cur.length }); cur.splice(i, 1); }
    const groups = RR.filter(x => x.g).map(x => x.g), at = n => cur.findIndex(c => c.names.length === 1 && c.names[0] === n);
    const flat = () => {
      const spans = groups.map(gr => [at(gr[0]), gr]).sort((a, b) => b[0] - a[0]);
      for (const [a, gr] of spans) { steps.push({ op: 'flatten', a, b: a + gr.length - 1, rank: cur.length }); cur.splice(a, gr.length, { names: gr.slice() }); }
    };
    const permute = keys => {
      const perm = keys.map(k => cur.findIndex(c => c.names.join(' ') === k));
      if (perm.every((v, i) => v === i)) return;
      const moved = perm.map((v, i) => (v !== i ? i : -1)).filter(i => i >= 0);
      if (moved.length === 2) steps.push({ op: 'transpose', a: moved[0], b: moved[1], rank: cur.length });
      else steps.push({ op: 'permute', perm, rank: cur.length });
      cur = perm.map(i => cur[i]);
    };
    // groups whose parts are already side by side are merged first (flatten, then move), else moved then merged
    const together = groups.every(gr => { const i0 = at(gr[0]); return i0 >= 0 && gr.every((n, k) => at(n) === i0 + k); });
    const keep = RR.filter(x => !x.one);
    if (together) { flat(); permute(keep.map(x => (x.g ? x.g.join(' ') : x.n))); }
    else { permute(keep.flatMap(x => (x.g ? x.g : [x.n]))); flat(); }
    RR.forEach((x, k) => { if (x.one) { steps.push({ op: 'unsqueeze', dim: k, rank: cur.length + 1 }); cur.splice(k, 0, { one: true }); } });
    return steps;
  }
  // one step as PyTorch; the last two axes are counted from the end (-2, -1), the others from the front.
  // step: inside a time loop, where axis 1 (time) is gone
  function reshapeStepCode(s, size, step) {
    const ix = (i, r) => (i >= r - 2 && r > 2 ? i - r : (step && i > 1 ? i - 1 : i));
    const n = (i, r) => (i < 0 ? i : ix(i, r));
    switch (s.op) {
      case 'unflatten': return `.unflatten(${s.dim < 0 ? s.dim : s.dim === s.rank - 1 ? -1 : n(s.dim, s.rank)}, (${s.sizes.map(v => (String(v) === '-1' ? '-1' : size(String(v)))).join(', ')}))`;
      case 'squeeze': return `.squeeze(${n(s.dim, s.rank)})`;
      case 'unsqueeze': return `.unsqueeze(${n(s.dim, s.rank)})`;
      case 'flatten': return s.b == null || s.b === s.rank - 1 || s.b === -1 ? `.flatten(${s.a < 0 ? s.a : step && s.a > 1 ? s.a - 1 : s.a})` : `.flatten(${n(s.a, s.rank)}, ${n(s.b, s.rank)})`;
      case 'transpose': { const both = s.a < 0 || (s.a >= s.rank - 2 && s.b >= s.rank - 2 && s.rank > 2); return both ? `.transpose(${s.a < 0 ? s.a : s.a - s.rank}, ${s.b < 0 ? s.b : s.b - s.rank})` : `.transpose(${step && s.a > 1 ? s.a - 1 : s.a}, ${step && s.b > 1 ? s.b - 1 : s.b})`; }
      case 'permute': return `.permute(${s.perm.join(', ')})`;
    }
    return '';
  }
  const dimPar = d => (/^[A-Za-z_]\w*$|^\d+$/.test(String(d)) ? String(d) : `(${d})`);
  // a product of sizes; h × (d/h) cancels back to d (merging what a split made)
  const dimMul = ds0 => {
    let ds = ds0.map(String);
    for (let i = 0; i < ds.length; i++) {
      const m = /^\(?(.+?)\)?\/(\(?[^/]+?\)?)$/.exec(ds[i]); if (!m) continue;
      const den = m[2].replace(/^\((.*)\)$/, '$1'), j = ds.findIndex((d, k) => k !== i && d.replace(/^\((.*)\)$/, '$1') === den);
      if (j >= 0) { ds[i] = m[1]; ds.splice(j, 1); i = -1; }
    }
    return ds.every(d => /^\d+$/.test(d)) ? String(ds.reduce((t, d) => t * +d, 1)) : ds.length === 1 ? ds[0] : ds.map(dimPar).join('*');
  };
  const dimDiv = (D, ks) => (ks.length ? (/^\d+$/.test(String(D)) && ks.every(k => /^\d+$/.test(String(k))) && +D % ks.reduce((t, k) => t * +k, 1) === 0 ? String(+D / ks.reduce((t, k) => t * +k, 1)) : `${dimPar(D)}/${ks.length > 1 ? `(${dimMul(ks)})` : dimPar(ks[0])}`) : String(D));
  // the output shape, and every axis's size by name (for the details)
  function reshapeShape(p, x) {
    if (Array.isArray(p._steps)) return { out: stepsShape(p._steps, x), val: {} };
    const { LL, RR, sizes } = reshapePlan(p, x.length), val = {};
    LL.forEach((it, i) => {
      const D = x[i];
      if (it.n) val[it.n] = D;
      else if (it.one) { if (/^\d+$/.test(String(D)) && String(D) !== '1') throw Error(`Axis ${i + 1} has size ${D}, not 1`); }
      else { const known = it.g.filter(n => sizes[n] != null), u = it.g.filter(n => sizes[n] == null); known.forEach(n => (val[n] = sizes[n])); if (u.length) val[u[0]] = dimDiv(D, known.map(n => sizes[n])); }
    });
    return { out: RR.map(it => (it.one ? '1' : it.n ? val[it.n] : dimMul(it.g.map(n => val[n])))), val, LL, sizes };
  }
  function stepsShape(steps, x) {
    let s = x.slice(); const nz = (i, r) => (i < 0 ? r + i : i);
    for (const st of steps) {
      if (st.op === 'unflatten') { const k = nz(st.dim, s.length), D = s[k], known = st.sizes.filter(v => String(v) !== '-1'); s.splice(k, 1, ...st.sizes.map(v => (String(v) === '-1' ? dimDiv(D, known) : String(v)))); }
      else if (st.op === 'flatten') { const a = nz(st.a, s.length), b = st.b == null ? s.length - 1 : nz(st.b, s.length); s.splice(a, b - a + 1, dimMul(s.slice(a, b + 1))); }
      else if (st.op === 'transpose') { const a = nz(st.a, s.length), b = nz(st.b, s.length); [s[a], s[b]] = [s[b], s[a]]; }
      else if (st.op === 'permute') s = st.perm.map(i => s[nz(i, s.length)]);
      else if (st.op === 'unsqueeze') s.splice(nz(st.dim, s.length + 1), 0, '1');
      else if (st.op === 'squeeze') { const k = nz(st.dim, s.length); if (String(s[k]) === '1') s.splice(k, 1); }
    }
    return s;
  }
  // for each output axis, the input axis it is (unchanged), or -1 (split, merged or new)
  function reshapeAxisSource(p, rIn) {
    if (Array.isArray(p._steps)) return null;
    const { LL, RR } = reshapePlan(p, rIn), from = new Map();
    LL.forEach((it, i) => { if (it.n) from.set(it.n, i); });
    return RR.map(it => (it.n && from.has(it.n) ? from.get(it.n) : -1));
  }
  // PyTorch steps read from code → a pattern, naming the axes after their sizes (e for the batch)
  function stepsToPattern(steps, x) {
    const used = new Set(), pool = 'abcdfgijklmnopqrsuvwxyz';
    const pick = (...cands) => { for (const c of cands) if (c && /^[a-z]$/.test(c) && !used.has(c)) { used.add(c); return c; } for (const c of pool) if (!used.has(c)) { used.add(c); return c; } return 'z' + used.size; };
    const sym = d => { const s = String(d); return /^[A-Za-z_]\w*$/.test(s) ? s : null; };
    const left = x.map(d => (String(d) === BATCH ? { n: pick('e') } : null));
    x.forEach((d, i) => { if (!left[i]) { const s = sym(d); left[i] = { n: s ? pick(s[0].toLowerCase(), (s.split('_').pop()[0] || '').toLowerCase()) : pick() }; } });
    let cur = left.map((it, i) => ({ el: [it.n], from: i })); const sizes = {};
    const nz = (i, r) => (i < 0 ? r + i : i);
    for (const st of steps) {
      if (st.op === 'unflatten') {
        const k = nz(st.dim, cur.length), c = cur[k];
        if (c.el.length === 1 && c.from != null && !left[c.from].g) {
          const orig = left[c.from].n; used.delete(orig);
          const nm = st.sizes.map(v => (String(v) === '-1' ? null : pick((sym(v) || '').split('_').pop()[0], (sym(v) || '')[0])));
          const fill = nm.map(n => n || pick(orig)); fill.forEach((n, j) => { if (String(st.sizes[j]) !== '-1') sizes[n] = String(st.sizes[j]); });
          left[c.from] = { g: fill }; cur.splice(k, 1, ...fill.map(n => ({ el: [n] })));
        } else if (c.el.length === st.sizes.length) cur.splice(k, 1, ...c.el.map(n => ({ el: [n] })));
        else return null;
      } else if (st.op === 'flatten') { const a = nz(st.a, cur.length), b = st.b == null ? cur.length - 1 : nz(st.b, cur.length); if (cur.slice(a, b + 1).some(c => c.one)) return null; cur.splice(a, b - a + 1, { el: cur.slice(a, b + 1).flatMap(c => c.el) }); }
      else if (st.op === 'transpose') { const a = nz(st.a, cur.length), b = nz(st.b, cur.length); [cur[a], cur[b]] = [cur[b], cur[a]]; }
      else if (st.op === 'permute') cur = st.perm.map(i => cur[nz(i, cur.length)]);
      else if (st.op === 'unsqueeze') cur.splice(nz(st.dim, cur.length + 1), 0, { one: true });
      else if (st.op === 'squeeze') {
        const k = nz(st.dim, cur.length), c = cur[k];
        if (c.one) cur.splice(k, 1);
        else if (c.el.length === 1 && c.from != null && !left[c.from].g && String(x[c.from]) === '1') { used.delete(left[c.from].n); left[c.from] = { one: true }; cur.splice(k, 1); }
        else if (String(x[c.from]) !== '1') { /* squeezing an axis that isn't 1 does nothing */ }
        else return null;
      }
    }
    const side = a => a.map(it => (it.one ? '1' : it.g ? `(${it.g.join(' ')})` : it.el ? (it.el.length > 1 ? `(${it.el.join(' ')})` : it.el[0]) : it.n)).join(' ');
    return { pattern: `${side(left)} -> ${side(cur)}`, sizes: Object.entries(sizes).map(([k, v]) => `${k}=${v}`).join(', ') };
  }
  // A reshape as the primitive nodes the variance engine simulates: split / merge heads as a whole, a swap of the last
  // two axes as Transpose, the rest as tensor methods. The last keeps the reshape's id; parent maps the others to it.
  function lowerReshapes(g) {
    if (!g.nodes.some(n => n.type === 'reshape')) return { graph: g, parent: new Map() };
    const { shapes } = inferShapes(g), ix = index(g), parent = new Map(), nodes = [], edges = g.edges.map(e => Object.assign({}, e));
    const nz = (i, R) => (i < 0 ? R + i : i);
    for (const n of g.nodes) {
      if (n.type !== 'reshape') { nodes.push(n); continue; }
      const x = shapes.get((ix.ins.get(n.id) || [])[0]) || [], r = x.length; let st = [];
      try { st = reshapeSteps(n.params, r); } catch { st = []; }
      const prim = [], pair = (a, b, R) => new Set([nz(a, R), nz(b, R)]);
      const is12 = (s2, R) => { const q = pair(s2.a, s2.b, R); return q.has(1) && q.has(2) && q.size === 2; };
      if (st.length === 2 && r === 3 && st[0].op === 'unflatten' && nz(st[0].dim, 3) === 2 && st[0].sizes.length === 2 && String(st[0].sizes[1]) === '-1' && st[1].op === 'transpose' && is12(st[1], 4)) prim.push({ type: 'heads', params: { heads: String(st[0].sizes[0]) } });
      else if (st.length === 2 && r === 4 && st[0].op === 'transpose' && is12(st[0], 4) && st[1].op === 'flatten' && nz(st[1].a, 4) === 2 && (st[1].b == null || nz(st[1].b, 4) === 3)) prim.push({ type: 'merge', params: {} });
      else {
        let R = r;
        for (const s2 of st) {
          if (s2.op === 'unflatten') { prim.push({ type: 'func', params: { fn: '.unflatten', extra: `${nz(s2.dim, R)}, (${s2.sizes.join(', ')})`, n: 1 } }); R += s2.sizes.length - 1; }
          else if (s2.op === 'flatten') { const a = nz(s2.a, R), b = s2.b == null ? R - 1 : nz(s2.b, R); prim.push({ type: 'func', params: { fn: '.flatten', extra: b === R - 1 ? `${a}` : `${a}, ${b}`, n: 1 } }); R -= b - a; }
          else if (s2.op === 'transpose') { const q = pair(s2.a, s2.b, R); prim.push(q.has(R - 1) && q.has(R - 2) ? { type: 'transpose', params: {} } : { type: 'func', params: { fn: '.transpose', extra: `${nz(s2.a, R)}, ${nz(s2.b, R)}`, n: 1 } }); }
          else if (s2.op === 'permute') prim.push({ type: 'func', params: { fn: '.permute', extra: s2.perm.map(i => nz(i, R)).join(', '), n: 1 } });
          else if (s2.op === 'unsqueeze') { prim.push({ type: 'func', params: { fn: '.unsqueeze', extra: String(nz(s2.dim, R + 1) - (R + 1)), n: 1 } }); R++; }
          else if (s2.op === 'squeeze') { prim.push({ type: 'func', params: { fn: '.squeeze', extra: String(nz(s2.dim, R)), n: 1 } }); R--; }
        }
      }
      if (!prim.length) prim.push({ type: 'func', params: { fn: 'identity', extra: '', n: 1 } });
      const ids = prim.map((_, i) => (i === prim.length - 1 ? n.id : `${n.id}~${i}`));
      prim.forEach((q, i) => { nodes.push({ id: ids[i], type: q.type, x: n.x, y: n.y, params: Object.assign({}, DEFAULTS[q.type] || {}, q.params, n.params.follows ? { follows: n.params.follows } : {}) }); if (i < prim.length - 1) parent.set(ids[i], n.id); });
      if (ids.length > 1) { for (const e of edges) if (e.to === n.id) e.to = ids[0]; for (let i = 1; i < ids.length; i++) edges.push({ id: `e~${ids[i]}`, from: ids[i - 1], to: ids[i], port: 0 }); }
    }
    return { graph: Object.assign({}, g, { nodes, edges }), parent };
  }
  // what a reshape does, in words: splits, merges, moves, new and removed axes
  function reshapeDescribe(p, x) {
    const r = reshapeShape(p, x), { LL, sizes, val } = r, said = [], outPlan = reshapePlan(p, x.length).RR;
    const show = n => (n.startsWith('…') ? '' : n), sz = n => val[n];
    LL.forEach((it, i) => { if (it.g) said.push({ k: 'split', axis: i, names: it.g, D: x[i], parts: it.g.map(sz) }); if (it.one) said.push({ k: 'drop', axis: i }); });
    outPlan.forEach((it, i) => { if (it.g) said.push({ k: 'merge', axis: i, names: it.g, parts: it.g.map(sz) }); if (it.one) said.push({ k: 'new', axis: i }); });
    const order = l => l.flatMap(it => (it.g ? it.g : it.n ? [it.n] : [])).filter(n => !n.startsWith('…'));
    const a = order(LL), b = order(outPlan);
    return { said, moved: a.join(' ') !== b.join(' '), before: a, after: b, sizes, val, show };
  }

  // Link sizes that must be equal (both inputs of an elementwise op) so they share one
  // named width: editing one then updates the other. Used after importing literal code.
  function linkSizes(g) {
    const { shapes } = inferShapes(g), ix = index(g);
    const parent = new Map(), find = k => { while (parent.get(k) !== k) { parent.set(k, parent.get(parent.get(k))); k = parent.get(k); } return k; };
    const add = k => { if (!parent.has(k)) parent.set(k, k); return k; };
    const union = (a, b) => { a = find(add(a)); b = find(add(b)); if (a !== b) parent.set(b, a); };
    // slot = the place a tensor's last size is set: a Linear's out, or an input's last dim
    // ax: which size, counted from the last (0). Followed back through layers that keep it, across a transpose or a
    // flatten (to the size it came from), to a convolution's channels
    const slotOf = (id, ax = 0) => {
      let n = ix.byId.get(id), guard = 0;
      while (n && guard++ < 500) {
        const rank = (shapes.get(n.id) || []).length;
        if (n.type === 'linear') return ax === 0 ? n.id + '|out' : null;
        if (n.type === 'embed') return ax === 0 ? n.id + '|dim' : null;
        if (n.type === 'conv') return rank === 4 && ax === 2 ? n.id + '|out' : null;
        if (n.type === 'input') { const r = parseShape(n.params.shape).length; return ax < r ? n.id + '|' + (r - 1 - ax) : null; }
        if (n.type === 'func' && n.params.init) return null;
        if (n.type === 'transpose') ax = ax === 0 ? 1 : ax === 1 ? 0 : ax;
        else if (n.type === 'reshape') {
          // an axis kept whole follows back to where it was; one that was split, merged or added has no single slot
          const src = ix.ins.get(n.id)[0], rIn = ((src && shapes.get(src)) || []).length; let map = null;
          try { map = reshapeAxisSource(n.params, rIn); } catch { map = null; }
          const j = map ? map[rank - 1 - ax] : -1; if (j == null || j < 0) return null; ax = rIn - 1 - j;
        }
        else if (n.type === 'func' && /(^|\.)flatten$/.test(n.params.fn)) {
          const src = ix.ins.get(n.id)[0], rIn = ((src && shapes.get(src)) || []).length, st = +(String(n.params.extra || '0').split(',')[0]) || 0, i = rank - 1 - ax;
          if (i >= (st < 0 ? rIn + st : st)) return null; ax = rIn - 1 - i;
        }
        else if (n.type === 'conv' || n.type === 'pool' || n.type === 'upsample') { if (ax !== 2) return null; }
        else if (n.type === 'func' && /(^|\.)unsqueeze$/.test(n.params.fn) && /^-?\d+$/.test(String(n.params.extra).trim())) {
          // the new size-1 dimension has no slot; the others move back by one past it
          const k0 = +n.params.extra, k = k0 < 0 ? rank + k0 : k0, i = rank - 1 - ax; if (i === k) return null; ax = (rank - 1) - 1 - (i > k ? i - 1 : i);
        }
        else if (n.type === 'func' && /(^|\.)(sum|mean|amax|prod)$/.test(n.params.fn) && /^-?\d+$/.test(String(n.params.extra).trim())) {
          const rIn = rank + 1, k0 = +n.params.extra, k = k0 < 0 ? rIn + k0 : k0, i = rank - 1 - ax; ax = rIn - 1 - (i >= k ? i + 1 : i);
        }
        if (n.type === 'param') return null;
        // a broadcast product or sum: follow the input that has this size (not a 1 broadcast up to it)
        let src = ix.ins.get(n.id)[0];
        if (n.type === 'binop' && !hasConst(n)) {
          const ok = s2 => { const sh = s2 && shapes.get(s2); return !!sh && ax < sh.length && String(sh[sh.length - 1 - ax]) !== '1'; };
          const [s0, s1] = ix.ins.get(n.id); if (!ok(s0) && ok(s1)) src = s1;
        }
        n = src ? ix.byId.get(src) : null;
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
        if (n.type === 'linear' || n.type === 'conv') { if (String(p.in) === String(oldDim)) { p.in = newDim; changed.push(n.id); } }
        else if (n.type === 'norm' || n.type === 'token') { if (String(p.dim) === String(oldDim)) { p.dim = newDim; changed.push(n.id); } walk(n.id); }
        else if (n.type === 'act' || n.type === 'dropout' || n.type === 'binop' || n.type === 'output' || n.type === 'pool' || n.type === 'upsample') walk(n.id);
      }
    };
    walk(nodeId);
    return changed;
  }

  // ── code generation ──
  const ACT_CODE = {
    sigmoid: x => `torch.sigmoid(${x})`, silu: x => `F.silu(${x})`, relu: x => `F.relu(${x})`,
    gelu: x => `F.gelu(${x})`, gelu_tanh: x => `F.gelu(${x}, approximate="tanh")`, tanh: x => `torch.tanh(${x})`, softplus: x => `F.softplus(${x})`,
  };
  const OP_PREC = { '+': 10, '-': 10, '*': 20, '/': 20, '@': 20 };
  // layers with weights of their own, which can be frozen (requires_grad_(False)); a block use can be frozen whole
  const FREEZABLE = new Set(['linear', 'conv', 'conv1d', 'embed', 'norm', 'token', 'param']);
  const DEF_ATTR = { linear: 'fc', norm: 'norm', dropout: 'drop', func: 'mod', embed: 'emb', conv: 'conv', token: 'cls_token', param: 'w', conv1d: 'conv1d' };
  const DEF_VAR = n => n.params.var || ({ linear: 'h', norm: 'n', act: 'a', dropout: 'h', func: 'y', embed: 'x', heads: 'h', merge: 'y', transpose: 't', reshape: 'x', scale: 's', mask: 'att', rope: 'r', conv: 'h', pool: 'p', upsample: 'u', token: 'x', select: 'x', param: 'w', conv1d: 'h', noise: 'eps' }[n.type] || (n.params.op === '*' ? 'g' : 'x'));
  const DIM_FIELDS = { linear: ['in', 'out'], norm: ['dim'], embed: ['num', 'dim'], heads: ['heads'], conv: ['in', 'out', 'k', 'stride', 'pad'], token: ['n', 'dim'], param: ['shape'], conv1d: ['ch', 'k'] };
  // [field, expression] for every size a node is built with; a block use's sizes are its args
  // a reshape's sizes (h = n_head) count as size fields by value; its axis names are local labels
  const sizePairs = str => String(str || '').split(/[,;]/).map(x => x.trim()).filter(Boolean).map(x => { const i = x.indexOf('='); return i < 0 ? null : [x.slice(0, i).trim(), x.slice(i + 1).trim()]; }).filter(Boolean);
  const mapSizes = (str, fn) => sizePairs(str).map(([k, v]) => `${k}=${fn(v)}`).join(', ');
  const dimList = n => (DIM_FIELDS[n.type] || []).map(f => [f, n.params[f]]).concat(n.type === 'reshape' ? sizePairs(n.params.sizes).map(([k, v]) => ['sizes.' + k, v]) : []).concat(n.type === 'block' ? Object.entries(n.params.args || {}).sort().map(([k, v]) => ['args.' + k, v]) : []);
  const baseOf = k => splitKey(String(k))[0];
  // small helper functions written above the classes when the model uses them
  const HELPERS = {
    heads: `def split_heads(x, n_head):\n    \"\"\"(B, T, C) -> (B, n_head, T, C // n_head)\"\"\"\n    return x.unflatten(-1, (n_head, -1)).transpose(1, 2)`,
    merge: `def merge_heads(x):\n    \"\"\"(B, n_head, T, D) -> (B, T, n_head * D)\"\"\"\n    return x.transpose(1, 2).flatten(2)`,
    rope: `def apply_rope(x, base=10000.0):\n    \"\"\"Rotary position embedding on (B, n_head, T, D): rotates feature pairs (i, i + D/2) by position-dependent angles.\"\"\"\n    t, d = x.size(-2), x.size(-1)\n    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)\n    ang = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), inv_freq)\n    ang = torch.cat((ang, ang), dim=-1)\n    x1, x2 = x[..., : d // 2], x[..., d // 2 :]\n    return x * ang.cos().to(x.dtype) + torch.cat((-x2, x1), dim=-1) * ang.sin().to(x.dtype)`,
    sincos_2d: `def sincos_2d(n, d):\n    \"\"\"Fixed 2-D sine/cosine position embedding of n patches on a square grid (I-JEPA, MAE): (n, d). A quarter of\n    the features each: sin and cos of the column, then of the row, at frequencies 1 / 10000^(i / (d/4)).\"\"\"\n    g, q = int(round(n ** 0.5)), d // 4\n    omega = 1.0 / 10000 ** (torch.arange(q, dtype=torch.float64) / q)\n    row, col = torch.arange(n, dtype=torch.float64).div(g, rounding_mode=\"floor\"), torch.arange(n, dtype=torch.float64) % g\n    a, b = col[:, None] * omega, row[:, None] * omega\n    return torch.cat([a.sin(), a.cos(), b.sin(), b.cos()], dim=1).float()`,
    causal_conv: `def causal_conv(conv, x):\n    \"\"\"A Conv1d along time on (B, T, C) that sees only the current and earlier steps (padding k - 1, cut back to T).\"\"\"\n    return conv(x.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)`,
    dt_bias: `def dt_bias_(bias, dt_min=0.001, dt_max=0.1):\n    \"\"\"Mamba's init of the time step's bias: softplus(bias) starts log-uniform in [dt_min, dt_max].\"\"\"\n    dt = (dt_min * (dt_max / dt_min) ** torch.rand_like(bias)).clamp(min=1e-4)\n    with torch.no_grad():\n        bias.copy_(dt + torch.log(-torch.expm1(-dt)))`,
    delay: `def delay(x):\n    \"\"\"One step later along time (dim 1): out[:, t] = x[:, t - 1], zeros at t = 0.\"\"\"\n    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)`,
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
  // an input that gives positions to take_positions (I-JEPA's masks): how many positions it picks from, else null
  function indexInputRange(g, ix, n, sym) {
    for (const o of ix.outs.get(n.id) || []) {
      const c = ix.byId.get(o.to); if (!c || c.type !== 'func' || c.params.fn !== 'take_positions' || o.port !== 1) continue;
      const src = (ix.inKeys.get(c.id) || [])[0], sh = src && inferShapes(g).shapes.get(src);
      if (sh && sh.length >= 2) return evalDim(sh[sh.length - 2], sym);
    }
    return null;
  }
  // name → { code, deps (other named ops it calls), shape(inS, sym) }
  const HELPER_OPS = {
    take_positions: { defaults: { n: 2, extra: '' }, deps: [], shape: inS => [inS[1][0], inS[1][1], inS[0][inS[0].length - 1]], code: String.raw`def take_positions(x, idx):
    """The positions idx (B, K) of x (B, N, D), or of a (1, N, D) table: (B, K, D), as I-JEPA's apply_masks."""
    return torch.gather(x.expand(idx.size(0), -1, -1), 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))` },
    last_positions: { defaults: { n: 2, extra: '' }, deps: [], shape: inS => [inS[0][0], inS[1][1], inS[0][inS[0].length - 1]], code: String.raw`def last_positions(x, like):
    """The last like.size(1) positions of x: what came of the tokens appended at the end (I-JEPA's predictions)."""
    return x[:, x.size(1) - like.size(1):]` },
    sinusoidal_positions: { defaults: { n: 1, extra: '' }, deps: [], shape: inS => inS[0], code: String.raw`def sinusoidal_positions(x, base=10000.0):
    """Add the fixed sine/cosine position encoding of "Attention Is All You Need" to x (B, T, D): feature 2i of
    position t gets sin(t / base^(2i/D)), feature 2i + 1 the cosine."""
    t, d = x.size(1), x.size(2)
    pos = torch.arange(t, device=x.device, dtype=torch.float32)[:, None]
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos * inv
    pe = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(-2)
    return x + pe.to(x.dtype)` },
    sinusoidal_pe: { defaults: { n: 1, extra: '' }, deps: [], shape: inS => inS[0].slice(-2), code: String.raw`def sinusoidal_pe(x, base=10000.0):
    """The fixed sine/cosine position encoding of "Attention Is All You Need" for x (B, T, D), as a (T, D) table to add
    to x: feature 2i of position t is sin(t / base^(2i/D)), feature 2i + 1 the cosine. Only x's sizes are used."""
    t, d = x.size(1), x.size(2)
    pos = torch.arange(t, device=x.device, dtype=torch.float32)[:, None]
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos * inv
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(-2).to(x.dtype)` },
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
      if (n.type === 'delay') want.add('delay');
      if (n.type === 'conv1d') want.add('causal_conv');
      if (n.type === 'param' && n.params.init === 'sincos2d') want.add('sincos_2d');
      if (n.type === 'linear' && n.params.bias && n.params.bias_init === 'dt') want.add('dt_bias');
      if (n.type === 'merge') want.add('merge');
      if (n.type === 'rope') want.add(n.params.mode === 'axial' ? 'ropeAxial' : n.params.mode === 'mrope' ? 'ropeMrope' : 'rope');
      if (n.type === 'mask') want.add(n.params.mode === 'given' ? 'maskGiven' : 'mask');
      if (n.type === 'func' && HELPER_OPS[n.params.fn]) { want.add(n.params.fn); HELPER_OPS[n.params.fn].deps.forEach(d => want.add(d)); }
    }
    const order = ['heads', 'merge', 'rope', 'ropeAxial', 'ropeMrope', 'mask', 'maskGiven', 'sincos_2d', 'causal_conv', 'dt_bias', 'delay', ...Object.keys(HELPER_OPS)];
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
            const keep = Object.keys(p).filter(k => !['name', 'title', 'note', 'tone', 'var', 'args', 'split', 'tie_x', 'sizes', ...(DIM_FIELDS[n.type] || [])].includes(k)).sort().map(k => k + '=' + (k === 'tie' && p[k] ? (localIdx.has(p[k]) ? 'm' + localIdx.get(p[k]) : 'out') : p[k]));
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
  const DIMF = { binop: ['const'], reshape: ['sizes'], linear: ['in', 'out'], norm: ['dim'], func: ['init'], input: ['shape'], embed: ['num', 'dim'], heads: ['heads'], conv: ['in', 'out', 'k', 'stride', 'pad'], token: ['n', 'dim'], param: ['shape'], conv1d: ['ch', 'k'] };
  function blockInputs(blk) { return blk.graph.nodes.filter(n => n.type === 'input'); }
  function blockOutputs(blk) { return blk.graph.nodes.filter(n => n.type === 'output'); }
  function blockParams(blk) {
    const fromInputs = new Set();
    for (const n of blockInputs(blk)) parseShape(n.params.shape).forEach(d => identsIn(d).forEach(t => fromInputs.add(t)));
    const ps = [];
    const exprs = n => (n.type === 'linear' ? [n.params.out] : n.type === 'conv' ? [n.params.out, n.params.k, n.params.stride, n.params.pad] : n.type === 'func' && n.params.init ? [n.params.init] : n.type === 'reshape' ? String(n.params.sizes || '').split(/[,;]/).map(x => x.split('=')[1] || '').filter(Boolean) : n.type === 'heads' ? [n.params.heads] : n.type === 'embed' ? [n.params.num, n.params.dim] : n.type === 'block' ? Object.values(n.params.args || {}).map(String) : []);
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
    for (const f of DIMF[node.type] || []) if ((f !== 'shape' || node.type === 'param') && typeof node.params[f] === 'string') node.params[f] = f === 'sizes' ? mapSizes(node.params[f], v => v.replace(IDENT, t => (map.has(t) ? map.get(t) : t))) : node.params[f].replace(IDENT, t => (map.has(t) ? map.get(t) : t));
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
        flat.groups.push({ id: reps > 1 ? `${n.id}#${r}` : n.id, frozen: !!n.params.frozen, follows: n.params.follows || '', momentum: n.params.momentum, momentum_end: n.params.momentum_end, cls: blk.name, attr: reps > 1 ? `${attr}_${r}` : attr, label: attr, nodes: members, argHints: hints, argDefaults: Object.assign({}, blk.graph.symbols),
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
      if (fn === 'kl') {
        // KL of the latent Gaussian: its mean and log-variance are both outputs of forward()
        const o2 = ts && outputs.find(z => ix.ins.get(z.id)[0] === ts);
        if (!o2) { warnings.push(`${LOSSES[fn].label}: connect its second input to the log-variance, a tensor that also goes to an Output.`); continue; }
        const P = outName(o), V = outName(o2), w = isFinite(+p.weight) && p.weight !== '' ? +p.weight : 1;
        terms.push({ id: L.id, expr: `${w === 1 ? '' : w + ' * '}kl_normal(${P}, ${V})`, doc: `KL divergence of N(${P}, exp(${V})) from N(0, 1)`, fn, out: P, outputId: o.id, target: null, targetName: null, mode: 'given' });
        continue;
      }
      const tn = ts && ix.byId.get(ts);
      if (!tn) { warnings.push(`${LOSSES[fn].label}: it has no target.`); continue; }
      // a target the model computes (I-JEPA: the target encoder's embeddings) is one of its outputs
      const tOut = tn.type !== 'input' && outputs.find(z => ix.ins.get(z.id)[0] === ts);
      if (tn.type !== 'input' && !tOut) { warnings.push(`${LOSSES[fn].label}: its target must be an Input arrow, or a tensor that also goes to an Output.`); continue; }
      // a loss on probabilities would apply softmax/sigmoid twice
      const pn = ix.byId.get(ps);
      if (pn && pn.type === 'act' && ((fn === 'cross_entropy' && pn.params.fn === 'softmax') || (fn === 'bce' && pn.params.fn === 'sigmoid'))) warnings.push(`${LOSSES[fn].label} takes raw scores (logits), but this output already goes through ${pn.params.fn}. Remove that ${pn.params.fn}: predict() applies it at inference.`);
      const P = outName(o), Ps = shapes.get(ps) || [];
      const T = tOut ? outName(tOut) : pyIdent(tn.params.name || 'target');
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
        } else if (Ps.length === 4) { addArg(T); expr = `F.cross_entropy(${P}, ${T}${k})`; doc = `cross-entropy of every pixel of ${P} against the class indices in ${T}`; }
        else { addArg(T); expr = Ps.length > 2 ? `F.cross_entropy(${P}.flatten(0, -2), ${T}.flatten()${k})` : `F.cross_entropy(${P}, ${T}${k})`; doc = `cross-entropy of ${P} against the class indices in ${T}`; }
      } else {
        if (!tOut) addArg(T);
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
    const imgOut = plan.outputs.map(o => ((plan.shapes.get(plan.ix.ins.get(o.id)[0]) || []).length === 4));
    const wrap = (a, x, i = 0) => (a === 'softmax' ? `torch.softmax(${x}, dim=${imgOut[i] ? 1 : -1})` : a === 'sigmoid' ? `torch.sigmoid(${x})` : x);
    const names = [...new Set(acts.filter(Boolean))];
    add.push('', '    @torch.no_grad()', `    def predict(self, ${fin.join(', ')}):`);
    add.push(`        ${Q}Inference: forward()${names.length ? ' followed by ' + names.join(' / ') + ', which the loss leaves out' : ' as is (the loss works on the raw output)'}.${Q}`);
    if (plan.outputs.length === 1) add.push(`        return ${wrap(acts[0], `self(${fin.join(', ')})`, 0)}`);
    else { add.push(`        outs = self(${fin.join(', ')})`); add.push(`        return ${plan.outputs.map((o, i) => wrap(acts[i], `outs[${i}]`, i)).join(', ')}`); }
    lines.splice(end, 0, ...add);
    for (const [k, ls] of r.lineMap) r.lineMap.set(k, ls.map(l => (l >= end ? l + add.length : l)));
    for (const [id, i] of lmap) r.lineMap.set(id, [end + i]);
    // the KL helper goes with the other helpers, above the first class
    if (plan.terms.some(t => t.fn === 'kl')) {
      const c1 = lines.findIndex(l => /^class /.test(l)), ins = [...KL_HELPER.split('\n'), '', ''];
      lines.splice(c1, 0, ...ins);
      for (const [k, ls] of r.lineMap) r.lineMap.set(k, ls.map(l => (l >= c1 ? l + ins.length : l)));
    }
    // smoke test: also compute the loss
    const st = lines.findIndex(l => /^if __name__/.test(l));
    if (st >= 0) {
      const { ix, shapes, sym } = plan, dimsOf = sh => sh.map(d => (d === BATCH ? sym[BATCH] || 8 : evalDim(d, sym) ?? 8));
      const demo = n => {
        const dims = dimsOf(parseShape(n.params.shape));
        const emb = ix.outs.get(n.id).map(o => ix.byId.get(o.to)).find(c => c && c.type === 'embed' && !c.params.positions);
        if (emb) return `torch.randint(0, ${evalDim(emb.params.num, sym) ?? 100}, (${dims.join(', ')}${dims.length === 1 ? ',' : ''}))`;
        const rng = indexInputRange(g, ix, n, sym); if (rng && dims.length === 2) return `torch.rand(${dims[0]}, ${rng}).argsort(dim=1)[:, :${dims[1]}]`;
        return `torch.randn(${dims.join(', ')})`;
      };
      const body = ['if __name__ == "__main__":', `    model = ${mainName}()`, '    print(sum(p.numel() for p in model.parameters()), "parameters")'];
      const lens = new Set(plan.fwdInputs.flatMap(n => { const d = parseShape(n.params.shape), tok = ix.outs.get(n.id).some(o => { const c = ix.byId.get(o.to); return c && ((c.type === 'embed' && !c.params.positions) || (c.type === 'func' && c.params.fn === 'take_positions' && o.port === 1)); }); return d.slice(1, tok ? d.length : -1).filter(x => /^[A-Za-z_]\w*$/.test(x)); }));
      const hint = n => (plan.fwdInputs.length > 1 && lens.size > 1 ? '  # (' + parseShape(n.params.shape).map(d => { const v = d === BATCH ? (sym[BATCH] || 8) : evalDim(d, sym); return /^[A-Za-z_]\w*$/.test(d) ? `${d}=${v ?? '?'}` : String(d); }).join(', ') + ')' : '');
      for (const n of plan.fwdInputs) body.push(`    ${pyIdent(n.params.name || 'x')} = ${n.params.demo || demo(n)}${hint(n)}`);
      const outN = plan.outputs.map(o => pyIdent(o.params.name || 'out'));
      body.push(`    ${outN.join(', ')} = model(${fin.join(', ')})`);
      for (const o of plan.outputs) { const sh = shapes.get(ix.ins.get(o.id)[0]); body.push(`    print(${pyIdent(o.params.name || 'out')}.shape)  # expected torch.Size([${(sh ? dimsOf(sh) : []).join(', ')}])`); }
      for (const a of plan.args.slice(plan.outputs.length)) {
        if (fin.includes(a)) continue;
        const t = plan.terms.find(z => z.targetName === a && z.fn !== 'kl');
        const P = t && shapes.get(ix.ins.get(t.outputId)[0]), pd = P ? dimsOf(P) : [8];
        const tup = d => `(${d.join(', ')}${d.length === 1 ? ',' : ''})`;
        const ex = !t ? 'None' : t.fn === 'cross_entropy' ? (pd.length === 4 ? `torch.randint(0, ${pd[1]}, ${tup([pd[0], pd[2], pd[3]])})` : `torch.randint(0, ${pd[pd.length - 1]}, ${tup(pd.slice(0, -1))})`) : t.fn === 'bce' ? `torch.randint(0, 2, ${tup(pd)}).float()` : `torch.randn(${pd.join(', ')})`;
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
      // a class's own constants go back to numbers in its layers and in the blocks it uses
      for (const gr of leaves) {
        const cs = gr.consts || {}; if (!Object.keys(cs).length) continue;
        const lit = e => String(e).replace(IDENT, t => (t in cs ? String(cs[t]) : t));
        for (const id of gr.nodes) {
          const n = g.nodes.find(z => z.id === id); if (!n) continue;
          for (const f of DIMF[n.type] || []) if ((f !== 'shape' || n.type === 'param') && typeof n.params[f] === 'string') n.params[f] = f === 'sizes' ? mapSizes(n.params[f], lit) : lit(n.params[f]);
          if (n.type === 'block' && n.params.args) for (const k of Object.keys(n.params.args)) n.params.args[k] = lit(n.params.args[k]);
        }
      }
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
    const getDim = (n, f) => (f.startsWith('args.') ? (n.params.args || {})[f.slice(5)] : f.startsWith('sizes.') ? (sizePairs(n.params.sizes).find(([k]) => k === f.slice(6)) || [])[1] : n.params[f]);
    for (const gi of ginfo) {
      const key = gi.group.cls + '|' + gi.ssig;
      // an image input (e × C × H × W, read by image layers) is sized by its channels, any other by its last size
      const readers = s => gi.order.filter(id => ix.inKeys.get(id).includes(s)).map(id => ix.byId.get(id));
      const seqOps = new Set(['linear', 'heads', 'merge', 'rope', 'mask', 'scale', 'transpose', 'embed', 'reshape']);
      const isImg = s => { const sh = shapes.get(s); return !!sh && sh.length === 4 && !readers(s).some(n => seqOps.has(n.type) || (n.type === 'norm' && !(NORMS[n.params.kind] || {}).image)); };
      const axisOf = s => { const sh = shapes.get(s); return sh ? (isImg(s) ? 1 : sh.length - 1) : -1; };
      const inDims = gi.extIn.map(s => { const sh = shapes.get(s); return sh ? sh[axisOf(s)] : null; });
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
          const D = inDims[j]; if (!D || nameOf.has(D) || (!/^[A-Za-z_]\w*$/.test(D) && !(isImg(s) && /^\d+$/.test(D)))) return;
          for (const id of gi.order) {
            if (!ix.inKeys.get(id).includes(s)) continue;
            const n = ix.byId.get(id), f = n.type === 'linear' || n.type === 'conv' ? n.params.in : n.type === 'norm' ? n.params.dim : null;
            if (f && /^[A-Za-z_]\w*$/.test(f) && nameOf.has(f) && evalDim(f, g.symbols) != null && evalDim(f, g.symbols) === evalDim(D, g.symbols)) { nameOf.set(D, nameOf.get(f)); break; }
          }
        });
        const inner = { name, symbols: { e: g.symbols.e || 8 }, nodes: [], edges: [], groups: [] };
        const idm = new Map();
        const inNodes = gi.extIn.map((s, j) => {
          const sh = shapes.get(s) || ['e', 'd'], ax = axisOf(s) < 0 ? sh.length - 1 : axisOf(s), img = isImg(s);
          const dims = sh.map((d, i) => { if (d === BATCH) return BATCH;
            // an image's height and width are H and W (H2, W2 for a second image input)
            if (img && i >= 2) { const nm = (i === 2 ? 'H' : 'W') + (j ? j + 1 : ''); if (!/^\d+$/.test(String(d)) && !nameOf.has(d)) nameOf.set(d, nm); inner.symbols[nm] = evalDim(d, g.symbols) ?? 32; return nm; }
            if (nameOf.has(d)) return nameOf.get(d);
            // a sequence's length is t, even when the caller's is a number (ViT's 197 = 196 patches + the class token)
            // a second sequence of another length (a decoder's memory from the encoder) is s
            if (/^\d+$/.test(String(d))) { if (!img && sh.length === 3 && i === 1 && i !== ax) { const nm = ['t', 's'].find(z => ![...nameOf.values()].includes(z)); if (nm) { nameOf.set(String(d), nm); return nm; } } return String(d); }
            if (i === ax) {
              const hv = byVal(d);
              // an image's channels are the one class argument worth the same, when that is already one of its sizes
              if (img && hv && [...nameOf].some(([t, v]) => v === hv && evalDim(t, g.symbols) === evalDim(d, g.symbols))) { nameOf.set(d, hv); return hv; }
              const nm = hv && ![...nameOf.values()].includes(hv) ? hv : inDims.length > 1 ? 'd' + (j + 1) : 'd'; nameOf.set(d, nm); return nm;
            }
            if (((sh.length === 3 && i === 1) || (sh.length === 2 && i === 0)) && d !== BATCH && ![...nameOf.values()].includes('t')) { nameOf.set(d, 't'); return 't'; } return d; });
          const z = makeNode('input', { name: gi.inNames ? gi.inNames[j] : gi.extIn.length > 1 ? 'x' + (j + 1) : 'x', shape: dims.join(', ') });
          inner.nodes.push(z); return z;
        });
        for (const [t, nm] of nameOf) inner.symbols[nm] = (gi.group.argDefaults && gi.group.argDefaults[nm] != null ? gi.group.argDefaults[nm] : evalDim(t, g.symbols)) ?? 64;
        for (const n of inner.nodes) parseShape(n.params.shape).forEach(d => identsIn(d).forEach(t => { if (inner.symbols[t] == null) inner.symbols[t] = g.symbols[t] ?? 16; }));
        for (const id of gi.order) {
          const c = JSON.parse(JSON.stringify(ix.byId.get(id))); c.id = uid('n');
          for (const f of DIMF[c.type] || []) if (typeof c.params[f] === 'string' && (f !== 'shape' || c.type === 'param')) c.params[f] = f === 'sizes' ? mapSizes(c.params[f], v => v.replace(IDENT, t => nameOf.get(t) || t)) : c.params[f].replace(IDENT, t => nameOf.get(t) || t);
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
      const use = makeNode('block', Object.assign({ block: ent.blk.id, args, nIn: gi.extIn.length, nOut: gi.outputs.length, name: gi.group.attr || '' }, gi.group.frozen ? { frozen: true } : {}), cx / gi.members.size, cy / gi.members.size);
      g.nodes.push(use); useOf.set(gi.group.id, use.id); gi.group.useId = use.id;
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
      if (!chained || arr.some(x => val(x.n) !== val(arr[0].n) || !!x.n.params.frozen !== !!arr[0].n.params.frozen)) continue;
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
    function emitModule(name, initSig, items, inputNames, outputRefs, map, inKeys) {
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
      // followers (an EMA tie): a layer or block use whose weights start as a copy of another's, get no gradients, run
      // without them, and are moved toward the other's by ema_update() after each training step
      const followInfo = it => (it.kind === 'node' ? (it.node.params.follows ? it.node.params : null) : it.kind === 'call' ? (it.gi.group.follows ? it.gi.group : null) : it.gis[0].group.follows ? it.gis[0].group : null);
      const itemOfUse = id => items.find(it => (it.kind === 'node' && it.node.id === id) || (it.kind === 'call' && String(it.gi.group.id) === id) || (it.kind === 'list' && it.gis.some(gi => String(gi.group.id).split('#')[0] === id)));
      const followers = [];
      for (const it of items) { const f = followInfo(it); if (!f) continue; const lead = itemOfUse(f.follows); if (lead && lead !== it) followers.push({ it, lead, m0: String(f.momentum ?? '0.996'), m1: String(f.momentum_end ?? '1.0') }); else warnings.push('A follower (EMA tie) has to follow a layer or block in the same module; that tie was left out.'); }
      const isFollower = new Set(followers.map(f => f.it.id));
      const isModCall = it => it.kind === 'call' || it.kind === 'list' || ['linear', 'norm', 'dropout', 'embed', 'conv', 'token', 'conv1d'].includes(it.node.type) || (it.node.type === 'func' && it.node.params.init);

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
      const isDelay = it => it && it.kind === 'node' && it.node.type === 'delay';
      const dfs = r => {
        r = baseRef(r);
        if (seen.has(r) || String(r).startsWith('in:') || !itemById.has(r)) return; seen.add(r);
        // a Delay reads its input from the previous step: what feeds it isn't needed first
        if (!isDelay(itemById.get(r))) for (const s of itemById.get(r).ins) dfs(s);
        order.push(r);
      };
      outputRefs.forEach(dfs);
      for (let k = 0; k < order.length; k++) { const it = itemById.get(order[k]); if (isDelay(it)) it.ins.forEach(dfs); }
      // ── recurrences: groups of items in a loop through a Delay (strongly connected), each run step by step ──
      const loopOf = new Map(), loops = [];
      {
        const idx = new Map(), low = new Map(), onSt = new Set(), st = []; let n0 = 0;
        const strong = v => {
          idx.set(v, n0); low.set(v, n0); n0++; st.push(v); onSt.add(v);
          for (const w0 of itemById.get(v).ins) { const w = baseRef(w0); if (!seen.has(w) || !itemById.has(w)) continue; if (!idx.has(w)) { strong(w); low.set(v, Math.min(low.get(v), low.get(w))); } else if (onSt.has(w)) low.set(v, Math.min(low.get(v), idx.get(w))); }
          if (low.get(v) === idx.get(v)) { const comp = []; let w; do { w = st.pop(); onSt.delete(w); comp.push(w); } while (w !== v); if (comp.length > 1 || itemById.get(v).ins.some(x => baseRef(x) === v)) { const L = { key: 'loop' + loops.length, members: new Set(comp) }; loops.push(L); comp.forEach(c => loopOf.set(c, L)); } }
        };
        for (const v of order) if (!idx.has(v)) strong(v);
      }
      // each loop's items together, after everything it reads and before everything that reads it
      if (loops.length) {
        const compOf = id => loopOf.get(id) || id, at = new Map(order.map((id, i) => [id, i]));
        const deps = new Map();
        for (const id of order) { const c = compOf(id); if (!deps.has(c)) deps.set(c, new Set()); for (const r0 of itemById.get(id).ins) { const r = baseRef(r0); if (at.has(r) && compOf(r) !== c) deps.get(c).add(compOf(r)); } }
        // among what is ready, the order the layers were drawn in
        const drawn = new Map(items.map((it, i) => [it.id, i]));
        const done = new Set(), out2 = [], comps = [...deps.keys()], first = c => (typeof c === 'string' ? drawn.get(c) ?? at.get(c) : Math.min(...[...c.members].map(m => drawn.get(m) ?? at.get(m))));
        comps.sort((a, b) => first(a) - first(b));
        while (out2.length < comps.length) { const c = comps.find(c2 => !done.has(c2) && [...deps.get(c2)].every(d => done.has(d))); if (!c) break; done.add(c); out2.push(c); }
        order.length = 0; for (const c of out2) { if (typeof c === 'string') order.push(c); else order.push(...[...c.members].sort((a, b) => at.get(a) - at.get(b))); }
      }
      const dead = items.filter(it => !seen.has(it.id) && !fused.has(it.id));
      if (dead.length) warnings.push(`${dead.length} layer(s) in ${name} don't reach an output and were left out.`);
      const uses = new Map(), consumer = new Map();
      const addUse = (r, by) => { uses.set(r, (uses.get(r) || 0) + 1); consumer.set(r, by); };
      for (const id of order) for (const s of itemById.get(id).ins) addUse(s, id);
      for (const id of order) { const it = itemById.get(id); if (it.kind === 'node' && (it.node.type === 'scale' || it.node.type === 'token' || (it.node.type === 'embed' && it.node.params.positions)) && it.ins[0]) addUse(it.ins[0], id); }
      outputRefs.forEach(r => addUse(r, '__out'));

      // init lines
      const init = [], initAt = new Map(), fwdAt = new Map();
      // sizes used inside forward() (e.g. the head count) are kept on self in named style: self.n_head = n_head
      const selfSyms = [];
      const fwdSize = expr => subst(expr, map).replace(IDENT, t => { if (!selfSyms.includes(t)) selfSyms.push(t); return 'self.' + t; });
      for (const it of items) if (it.kind === 'node' && it.node.type === 'heads') fwdSize(it.node.params.heads);
      for (const it of items) if (it.kind === 'node' && it.node.type === 'reshape') { try { Object.values(parseSizes(it.node.params.sizes)).forEach(v => { if (!/^-?\d+$/.test(v)) fwdSize(v); }); } catch { /* reported by the shapes */ } }
      // a constant written with sizes (x * d ** 0.5) reads them from self in forward()
      const constCode = new Map(), isSym = t => (g.symbols && g.symbols[t] != null) || map.has(t);
      for (const it of items) if (it.kind === 'node' && it.node.type === 'binop' && hasConst(it.node)) { const c = String(it.node.params.const); const ids = c.match(IDENT) || []; if (ids.length && ids.every(isSym)) constCode.set(it.id, c.replace(IDENT, t => (map && map.has(t) ? map.get(t) : t)).replace(IDENT, t => { if (!selfSyms.includes(t)) selfSyms.push(t); return 'self.' + t; })); }
      // a loop's state starts as zeros of its per-step size
      for (const it of items) if (isDelay(it) && loopOf.has(it.id)) (shapes.get(it.node.id) || []).slice(2).forEach(d => { if (!/^\d+$/.test(String(subst(d, map)))) fwdSize(d); });
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
          if (it.gis.every(gi => gi.group.frozen) && !isFollower.has(it.id)) init.push(`self.${a}.requires_grad_(False)`);
          initAt.set(id, [start, init.length]);
          continue;
        }
        if (it.kind === 'call') {
          const c = it.gi.cls; init.push(`self.${attrFor(it)} = ${c.name}(${it.gi.ids.map(t => subst(t, map)).join(', ')})`);
          if (it.gi.group.frozen && !isFollower.has(it.id)) init.push(`self.${attrFor(it)}.requires_grad_(False)`);
          continue;
        }
        const n = it.node, p = n.params, S = e => subst(e, map);
        if (n.type === 'linear') {
          const a = attrFor(it);
          init.push(`self.${a} = nn.Linear(${S(p.in)}, ${S(p.out)}${p.bias ? '' : ', bias=False'})`);
          const ic = p.tie ? null : initCode(p, `self.${a}.weight`);
          if (ic) init.push(ic);
          // a tied layer's weight comes from elsewhere, but its own bias is still zeroed when asked
          if (p.bias && (p.tie ? p.zero_bias === true : ic && p.zero_bias !== false)) init.push(`nn.init.zeros_(self.${a}.bias)`);
          if (p.bias && p.bias_init === 'dt') init.push(`dt_bias_(self.${a}.bias)`);
        }
        else if (n.type === 'norm') init.push(`self.${attrFor(it)} = ${normCtor(p, S(p.dim))}`);
        else if (n.type === 'conv') {
          // nn.Conv2d(c_in, c_out, k, stride=s, padding=p): stride and padding only when they aren't the defaults
          const st = S(p.stride), pd = S(p.pad);
          init.push(`self.${attrFor(it)} = nn.${p.transposed ? 'ConvTranspose2d' : 'Conv2d'}(${S(p.in)}, ${S(p.out)}, ${S(p.k)}${st !== '1' ? `, stride=${st}` : ''}${pd !== '0' ? `, padding=${pd}` : ''}${p.bias ? '' : ', bias=False'})`);
        }
        else if (n.type === 'dropout') init.push(`self.${attrFor(it)} = nn.Dropout(${p.p})`);
        else if (n.type === 'token') init.push(`self.${attrFor(it)} = nn.Parameter(torch.zeros(1, ${S(p.n)}, ${S(p.dim)}))`);
        else if (n.type === 'param') {
          const dims = parseShape(p.shape).map(x => S(x)), N = dims[dims.length - 1], par = x => (/^[A-Za-z_]\w*$|^\d+$/.test(x) ? x : `(${x})`);
          const v = p.init === 'sincos2d' ? `sincos_2d(${dims[dims.length - 2]}, ${dims[dims.length - 1]})${dims.length > 2 ? '.reshape(' + dims.join(', ') + ')' : ''}` : p.init === 's4d' ? `torch.log(torch.arange(1, ${/^\d+$/.test(N) ? +N + 1 : par(N) + ' + 1'}, dtype=torch.float32))${dims.length > 1 ? `.repeat(${dims.slice(0, -1).join(', ')}, 1)` : ''}`
            : `torch.${p.init === 'ones' ? 'ones' : 'zeros'}(${dims.join(', ')})`;
          init.push(`self.${attrFor(it)} = nn.Parameter(${v})`);
          if (p.init === 'normal') init.push(`nn.init.normal_(self.${attrFor(it)}, std=${p.init_arg || '0.02'})`);
        }
        else if (n.type === 'conv1d') {
          const c = S(p.ch), k = S(p.k), pd = /^\d+$/.test(k) ? String(+k - 1) : `${k} - 1`;
          init.push(`self.${attrFor(it)} = nn.Conv1d(${c}, ${c}, ${k}, groups=${c}, padding=${pd}${p.bias ? '' : ', bias=False'})`);
        }
        else if (n.type === 'embed') { const a = attrFor(it); init.push(`self.${a} = nn.Embedding(${S(p.num)}, ${S(p.dim)})`); const ic = p.tie ? null : initCode(p, `self.${a}.weight`); if (ic) init.push(ic); }
        else if (n.type === 'func' && p.init) init.push(`self.${attrFor(it)} = ${S(p.init)}`);
        // frozen: its weights get no updates (the optimizer only sees trainable ones)
        if (p.frozen && !isFollower.has(it.id) && FREEZABLE.has(n.type) && init.length > start) init.push(`self.${attrFor(it)}.requires_grad_(False)`);
        if (init.length > start) initAt.set(id, [start, init.length]);
      }
      // tied weights: one Parameter shared by two layers (set after both exist)
      for (const id of order) {
        const it = itemById.get(id); if (!it || it.kind !== 'node' || !it.node.params.tie) continue;
        const o = itemById.get(it.node.params.tie);
        if (!o || o.kind !== 'node' || !weightShape(o.node)) { warnings.push(`${attrFor(it)}: its tied layer isn't in the same module, so the tie was left out.`); continue; }
        const start = init.length; init.push(`self.${attrFor(it)}.weight = self.${attrFor(o)}.weight`);
        initAt.set(id + ':tie', [start, init.length]);
        if (!!it.node.params.frozen !== !!o.node.params.frozen) warnings.push(`${attrFor(it)} shares its weight with ${attrFor(o)}: the shared weight is frozen only if ${attrFor(o)} is. Freeze both, or neither.`);
      }
      // followers start as a copy of what they follow, and get no gradients
      for (const f of followers) {
        const start = init.length, A = attrFor(f.it), B = attrFor(f.lead);
        init.push(`self.${A}.load_state_dict(self.${B}.state_dict())`, `self.${A}.requires_grad_(False)`);
        initAt.set('follow:' + f.it.id, [start, init.length]);
      }

      // forward
      const lines = [], expr = new Map();
      const feedsLoop = new Set(); for (const L of loops) for (const m of L.members) for (const r of itemById.get(m).ins) if (r && !L.members.has(baseRef(r))) feedsLoop.add(baseRef(r));
      const inputIds = inKeys || [];
      const live = new Map(); // var -> remaining uses
      const varOf = new Map();
      inputNames.forEach((nm, k) => { live.set(nm, uses.get('in:' + k) || 0); varOf.set('in:' + k, nm); });
      const ref = r => {
        if (varOf.has(r)) return { code: varOf.get(r), prec: 100, refs: [varOf.get(r)], ids: [] };
        return expr.get(r);
      };
      const wrap = (e, prec, right) => (e.prec < prec || (right && e.prec === prec) ? `(${e.code})` : e.code);
      // a loop's state keeps its name from the diagram (h, c): other values don't take it
      const reserved = new Set(); for (const L of loops) for (const m of L.members) { const it = itemById.get(m); if (isDelay(it)) reserved.add(pyIdent(it.node.params.name || 'h')); }
      const alloc = (base, own) => {
        base = pyIdent(base);
        const free = v => !live.get(v) && (own || !reserved.has(v));
        if (free(base)) return base;
        for (let k = 2; ; k++) if (free(base + k)) return base + k;
      };
      const release = refs => { for (const v of refs) if (live.has(v)) live.set(v, live.get(v) - 1); };
      for (const id of order) {
        if (loopOf.has(id)) { const L = loopOf.get(id); if (!L.emitted) { L.emitted = true; emitLoop(L); } continue; }
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
          const ng = isFollower.has(id) ? '    ' : ''; if (ng) { mark(ids, lines.length); lines.push('with torch.no_grad():'); }
          mark(ids, lines.length);
          const cm = opts.shapeComments ? `  # ${it.gi.outputs.map(o => shapeStr(o, map)).join(', ')}` : '';
          lines.push(`${ng}${vs.join(', ')} = self.${attrFor(it)}(${a.map(x => x.code).join(', ')})` + cm);
          continue;
        }
        if (it.kind === 'list') {
          const ids = [id, ...a.flatMap(x => x.ids || [])];
          // the shared inputs stay live through the loop: the loop variable can't take one of their names
          release(a[0].refs);
          const v = alloc('x');
          release(a.slice(1).flatMap(x => x.refs));
          if (a[0].code !== v) { mark(ids, lines.length); lines.push(`${v} = ${a[0].code}`); }
          const ng = isFollower.has(id) ? '    ' : ''; if (ng) { mark(ids, lines.length); lines.push('with torch.no_grad():'); }
          mark(ids, lines.length);
          lines.push(`${ng}for blk in self.${attrFor(it)}:` + (opts.shapeComments ? `  # ${shapeStr(it.gis[it.gis.length - 1].output, map)}` : ''));
          mark(ids, lines.length); lines.push(`${ng}    ${v} = blk(${[v, ...a.slice(1).map(x => x.code)].join(', ')})`);
          live.set(v, uses.get(id) || 0); varOf.set(id, v);
          continue;
        }
        let refs = a.flatMap(x => x.refs);
        // the input appears twice in these expressions, so its variable is used twice
        if (it.kind === 'node' && (it.node.type === 'scale' || it.node.type === 'token' || (it.node.type === 'embed' && it.node.params.positions)) && a[0]) refs = refs.concat(a[0].refs);
        const { code, prec } = nodeCode(it, a);
        const e = { code, prec, refs, ids: [id, ...a.flatMap(x => x.ids || [])], hasCall: isModCall(it) || a.some(x => x.hasCall), top: it.kind === 'node' && it.node.type === 'binop' ? 'op' : 'fn' };
        const u = uses.get(id) || 0, c = consumer.get(id), cit = itemById.get(c);
        // residual adds stay on their own line; a single-use value can be folded into its consumer (always into return)
        const isAdd = it.kind === 'node' && it.node.type === 'binop' && it.node.params.op === '+' && it.ins.length === 2;
        // a layer's result that feeds another layer gets its own line (x = F.relu(self.fc1(x)), not nested calls)
        const nests = cit && isModCall(cit) && e.hasCall && e.top !== 'op';
        const named = it.kind === 'node' && it.node.params.var && c !== '__out'; // a value named in the diagram gets its own line
        // what a loop reads gets a line of its own before the loop (it is sliced at every step)
        const isParam = it.kind === 'node' && it.node.type === 'param' && !it.node.params.var;
        const rr = it.kind === 'node' && it.node.type === 'reshape' && cit && cit.kind === 'node' && cit.node.type === 'reshape';
        const inline = !rr && (isParam || !named && !isFollower.has(id) && !feedsLoop.has(id) && u === 1 && !(cit && cit.kind === 'list') && (c === '__out' || (code.length <= 60 && !nests && !(isModCall(it) && cit && isModCall(cit)) && !(isAdd && cit && isModCall(cit)))));
        if (inline) { expr.set(id, e); continue; }
        release(refs);
        const v = alloc(it.kind === 'node' ? DEF_VAR(it.node) : 'x');
        live.set(v, u); varOf.set(id, v);
        const sid = it.kind === 'call' ? it.gi.output : it.node.id;
        const ng = isFollower.has(id) ? '    ' : ''; if (ng) { mark(e.ids, lines.length); lines.push('with torch.no_grad():'); }
        mark(e.ids, lines.length);
        lines.push(`${ng}${v} = ${code}` + (opts.shapeComments ? `  # ${shapeStr(sid, map)}` : ''));
      }
      // the code of one item from its arguments' code (step: inside a time loop, where the time axis is gone)
      function nodeCode(it, a, step) {
        let code, prec = 100;
        if (it.kind === 'call') code = `self.${attrFor(it)}(${a.map(x => x.code).join(', ')})`;
        else {
          const n = it.node, p = n.params;
          if (it.silu) code = `F.silu(${a[0].code})`;
          else switch (n.type) {
            case 'delay': code = `delay(${a[0].code})`; break;
            case 'token': code = `torch.cat([self.${attrFor(it)}.expand(${wrap(a[0], 100)}.size(0), -1, -1), ${a[0].code}], dim=1)`; break;
            case 'select': code = `${wrap(a[0], 100)}[:, ${p.index}]`; break;
            case 'param': code = `self.${attrFor(it)}`; break;
            case 'noise': code = `torch.randn_like(${a[0].code})`; break;
            case 'conv1d': code = `causal_conv(self.${attrFor(it)}, ${a[0].code})`; break;
            case 'norm': {
              const rank = (shapes.get(ix.ins.get(n.id)[0]) || []).length - (step ? 1 : 0);
              code = NORMS[p.kind] && NORMS[p.kind].channelFirst && rank === 3
                ? `self.${attrFor(it)}(${wrap(a[0], 100)}.transpose(1, 2)).transpose(1, 2)`
                : `self.${attrFor(it)}(${a[0].code})`;
              break;
            }
            case 'linear': case 'dropout': case 'conv': code = `self.${attrFor(it)}(${a[0].code})`; break;
            case 'pool': {
              if (p.kind === 'adaptive') { code = `F.adaptive_avg_pool2d(${a[0].code}, ${p.size || 1})`; break; }
              // F.max_pool2d(x, k[, stride[, padding]]): stride defaults to k
              const st = p.stride !== '' && p.stride != null && String(p.stride) !== String(p.k) ? String(p.stride) : '', pd = String(p.pad || '0') !== '0' ? String(p.pad) : '';
              code = `F.${p.kind === 'avg' ? 'avg' : 'max'}_pool2d(${a[0].code}, ${p.k}${st || pd ? ', ' + (st || p.k) : ''}${pd ? ', ' + pd : ''})`;
              break;
            }
            case 'upsample': code = `F.interpolate(${a[0].code}, scale_factor=${p.scale}, mode="${p.mode || 'nearest'}")`; break;
            case 'embed': code = p.positions ? `self.${attrFor(it)}(torch.arange(${wrap(a[0], 100)}.size(1), device=${wrap(a[0], 100)}.device))` : `self.${attrFor(it)}(${a[0].code})`; break;
            case 'heads': code = `split_heads(${a[0].code}, ${fwdSize(p.heads)})`; break;
            case 'merge': code = `merge_heads(${a[0].code})`; break;
            case 'transpose': code = `${wrap(a[0], 100)}.transpose(-2, -1)`; break;
            case 'reshape': {
              const x = shapes.get(ix.ins.get(n.id)[0]) || [];
              let st; try { st = reshapeSteps(p, x.length); } catch { st = []; }
              code = `${wrap(a[0], 100)}${st.map(s2 => reshapeStepCode(s2, v => (/^-?\d+$/.test(v) ? v : fwdSize(v)), step)).join('')}`; break;
            }
            case 'scale': prec = 20; code = `${wrap(a[0], 20)} * ${wrap(a[0], 100)}.size(-1) ** -0.5`; break;
            case 'mask': code = p.mode === 'given' ? `apply_mask(${a[0].code}, ${a[1].code})` : `causal_mask(${a[0].code})`; break;
            case 'rope': {
              const base = +p.base === 10000 ? '' : `, ${p.base}`, secs = String(p.sections || '16, 24, 24').split(',').map(v => v.trim()).filter(Boolean);
              if (p.mode === 'axial') code = `apply_rope_axial(${a[0].code}, ${a[1].code}${base})`;
              else if (p.mode === 'mrope') code = `apply_rope_mrope(${a[0].code}, ${a[1].code}, ${p.base}${secs.join(', ') === '16, 24, 24' ? '' : `, (${secs.join(', ')})`})`;
              else code = `apply_rope(${a[0].code}${base})`;
              break;
            }
            case 'act': code = p.fn === 'softmax' ? `torch.softmax(${a[0].code}, dim=${p.dim ?? -1})` : p.fn === 'leaky_relu' ? `F.leaky_relu(${a[0].code}${String(p.slope ?? '0.01') === '0.01' ? '' : ', ' + p.slope})` : (ACT_CODE[p.fn] || ACT_CODE.silu)(a[0].code); break;
            case 'binop': {
              const op = p.op || '+'; prec = OP_PREC[op] || 10;
              const cst = constCode.has(it.id) ? constCode.get(it.id) : p.const;
              if (a.length === 1 && op === '*' && !p.rev && String(p.const).trim() === '-1') { prec = 25; code = `-${wrap(a[0], 26)}`; }
              else if (a.length === 1) code = p.rev ? `${cst} ${op} ${wrap(a[0], prec, true)}` : `${wrap(a[0], prec)} ${op} ${cst}`;
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
        return { code, prec };
      }
      // ── a recurrence: one for loop over time ──
      // Items outside the loop that it reads are sliced at step t (when they have the time axis); each Delay is a state
      // variable that starts as zeros and takes its input's value at the end of every step; what the rest of the model
      // reads from the loop is collected and stacked back along time.
      function emitLoop(L) {
        // the step's lines in the order the layers were drawn, each after what it reads
        const mem = order.filter(id => L.members.has(id)), body = [];
        { const drawn = new Map(items.map((it, i) => [it.id, i])), left = mem.filter(id => !isDelay(itemById.get(id))).sort((a, b) => drawn.get(a) - drawn.get(b));
          while (left.length) { const k = left.findIndex(id => itemById.get(id).ins.every(r => { const b = baseRef(r); return !left.includes(b) || b === id; })); body.push(...left.splice(k < 0 ? 0 : k, 1)); } }
        const delays = mem.filter(id => isDelay(itemById.get(id)));
        const timeDim = (shapes.get(itemById.get(delays[0]).node.id) || [])[1];
        const extRefs = [...new Set(body.flatMap(id => itemById.get(id).ins).filter(r => r && !L.members.has(baseRef(r))))];
        const seq = r => { const it2 = itemById.get(baseRef(r)), sid = it2 ? (it2.kind === 'call' ? it2.gi.output : it2.node && it2.node.id) : String(r).startsWith('in:') ? inputIds[+String(r).slice(3)] : null; const sh = sid && shapes.get(sid); return !!sh && sh.length >= 3 && String(sh[1]) === String(timeDim); };
        const anchor = extRefs.find(seq);
        if (!anchor) { warnings.push(`A loop through a Delay needs a sequence coming in from outside it; it was left out.`); return; }
        const A = ref(anchor), Ac = wrap(A, 100), allIds = [...mem];
        const usedOutside = id => (uses.get(id) || 0) > [...L.members].reduce((t, m) => t + itemById.get(m).ins.filter(r => baseRef(r) === id).length, 0);
        // state variables, named as in the diagram (h, c)
        const stateOf = new Map();
        for (const d of delays) { const nm = alloc(itemById.get(d).node.params.name || 'h', true); live.set(nm, 1e9); stateOf.set(d, nm); }
        for (const d of delays) {
          const dims = (shapes.get(itemById.get(d).node.id) || []).slice(2).map(x => { const v = subst(x, map); return /^\d+$/.test(String(v)) ? v : fwdSize(x); });
          mark([d], lines.length); lines.push(`${stateOf.get(d)} = ${Ac}.new_zeros(${Ac}.size(0)${dims.map(x => ', ' + x).join('')})`);
        }
        // the producer of a Delay's input can write the state directly when nothing reads the old state after it
        const pos = new Map(body.map((id, i) => [id, i]));
        const writer = new Map();
        for (const d of delays) {
          const src = baseRef(itemById.get(d).ins[0]); if (!pos.has(src)) continue;
          const readers = body.filter(id => itemById.get(id).ins.some(r => baseRef(r) === d));
          if (readers.every(r => pos.get(r) <= pos.get(src)) && !writer.has(src)) writer.set(src, d);
        }
        const collect = body.filter(usedOutside), lists = new Map();
        for (const id of collect) { const it2 = itemById.get(id); const nm = alloc((writer.has(id) ? stateOf.get(writer.get(id)) : it2.kind === 'node' ? DEF_VAR(it2.node) : 'x') + 's'); live.set(nm, 1e9); lists.set(id, nm); mark([id], lines.length); lines.push(`${nm} = []`); }
        const tv = alloc('t'); live.set(tv, 1e9);
        mark(allIds, lines.length); lines.push(`for ${tv} in range(${Ac}.size(1)):`);
        const stepVar = new Map(), stepExpr = new Map(), bodyLines = [];
        const sref = r => {
          const b = baseRef(r);
          if (stepVar.has(r)) return { code: stepVar.get(r), prec: 100, ids: [] };
          if (stepExpr.has(b)) return stepExpr.get(b);
          if (stateOf.has(b)) return { code: stateOf.get(b), prec: 100, ids: [b] };
          const o = ref(r); if (!o) return null;
          return seq(r) ? { code: `${wrap(o, 100)}[:, ${tv}]`, prec: 100, ids: o.ids || [] } : o;
        };
        const inUses = id => body.reduce((t, m) => t + itemById.get(m).ins.filter(r => baseRef(r) === id).length, 0);
        const late = [];
        for (const id of body) {
          const it2 = itemById.get(id), args = it2.ins.map(sref);
          if (args.some(x => !x)) continue;
          const { code, prec } = nodeCode(it2, args, true);
          const ids = [id, ...args.flatMap(x => x.ids || [])];
          const cons = body.filter(m => itemById.get(m).ins.some(r => baseRef(r) === id)), cit2 = cons.length === 1 ? itemById.get(cons[0]) : null;
          const fold = !writer.has(id) && !lists.has(id) && !delays.some(d => baseRef(itemById.get(d).ins[0]) === id) && inUses(id) === 1 && code.length <= 60 && !(isModCall(it2) && cit2 && isModCall(cit2)) && !(it2.kind === 'node' && it2.node.params.var);
          if (fold) { stepExpr.set(id, { code, prec, ids }); continue; }
          let v;
          if (writer.has(id)) v = stateOf.get(writer.get(id));
          else { v = alloc(it2.kind === 'node' ? DEF_VAR(it2.node) : 'x'); live.set(v, 1e9); }
          stepVar.set(id, v);
          const sid = it2.kind === 'call' ? it2.gi.output : it2.node.id, sh = shapes.get(sid);
          bodyLines.push([ids, `${v} = ${code}` + (opts.shapeComments && sh ? `  # (${sh.filter((_, i) => i !== 1).map(d => subst(d, map)).join(', ')})` : '')]);
          if (lists.has(id)) bodyLines.push([[id], `${lists.get(id)}.append(${v})`]);
        }
        // states whose new value isn't written in place: set at the end of the step
        for (const d of delays) { const src = baseRef(itemById.get(d).ins[0]); if (writer.get(src) === d) continue; const o = sref(itemById.get(d).ins[0]); if (o) late.push([[d], `${stateOf.get(d)} = ${o.code}`]); }
        for (const [ids, l] of [...bodyLines, ...late]) { mark(ids, lines.length); lines.push('    ' + l); }
        // after the loop: the collected steps, stacked back along time
        for (const id of collect) {
          const it2 = itemById.get(id), v = writer.has(id) ? stateOf.get(writer.get(id)) : stepVar.get(id);
          mark([id], lines.length); lines.push(`${v} = torch.stack(${lists.get(id)}, dim=1)` + (opts.shapeComments ? `  # ${shapeStr(it2.kind === 'call' ? it2.gi.output : it2.node.id, map)}` : ''));
          varOf.set(id, v); live.set(v, (uses.get(id) || 0) - inUses(id));
        }
        for (const v of [...stepVar.values(), ...stateOf.values(), ...lists.values(), tv]) if (![...collect].some(id => varOf.get(id) === v)) live.set(v, 0);
        release(extRefs.flatMap(r => (ref(r) || {}).refs || []));
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
        ...(followers.length ? ['', '    @torch.no_grad()', '    def ema_update(self, progress):',
          `        ${'"'.repeat(3)}Move each follower toward the weights it follows (an exponential moving average); call after each${'\n'}        optimizer step with progress, the share of training done (0 → 1), which sets the momentum m.${'"'.repeat(3)}`,
          ...followers.flatMap(f => { const A = attrFor(f.it), B = attrFor(f.lead);
            return [`        m = ${f.m0 === f.m1 ? f.m0 : `${f.m0} + (${f.m1} - ${f.m0}) * progress`}  # ${A} follows ${B}`,
              `        for t, s in zip(self.${A}.parameters(), self.${B}.parameters()):`, '            t.lerp_(s, 1 - m)']; })] : []),
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
      const off = offset(), m = emitModule(c.name, sig2, items, extNames, gi.outputs.map(o => (gi.members.has(o) ? o : 'in:' + gi.extIn.indexOf(o))), map, gi.extIn);
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
      if (n.type === 'reshape') { try { Object.values(parseSizes(n.params.sizes)).forEach(note); } catch { /* reported by the shapes */ } }
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
    const off = offset(), m = emitModule(mainName, sig, items, inputs.map(n => pyIdent(n.params.name || 'x')), outRefs, opts.literalDims ? LIT : null, inputs.map(n => n.id));
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
      // positions for take_positions: distinct ones, drawn from the sequence it picks from
      const rng = indexInputRange(g, ix, n, sym); if (rng && dims.length === 2) return `torch.rand(${dims[0]}, ${rng}).argsort(dim=1)[:, :${dims[1]}]`;
      return `torch.randn(${dims.join(', ')})`;
    });
    // an input can bring its own example (an image grid, ids with image tokens): then every input gets a line of
    // its own, with its shape and sizes in a comment that reads back exactly
    // (also when inputs have sequence lengths of their own, like a translation model's source s and target t)
    const lens = new Set(inputs.flatMap(n => { const d = parseShape(n.params.shape), tok = ix.outs.get(n.id).some(o => { const c = ix.byId.get(o.to); return c && ((c.type === 'embed' && !c.params.positions) || (c.type === 'func' && c.params.fn === 'take_positions' && o.port === 1)); }); return d.slice(1, tok ? d.length : -1).filter(x => /^[A-Za-z_]\w*$/.test(x)); }));
    const custom = inputs.some(n => n.params.demo) || (inputs.length > 1 && lens.size > 1);
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
    if (usedHelpers.includes(HELPERS.delay) && !out.some(t => t !== HELPERS.delay && /(^|[^\w.])delay\(/.test(t.replace(HELPERS.delay, '')))) {
      const k = out.findIndex(t => t.includes(HELPERS.delay)), before = out.join('\n').split('\n').length;
      out[k] = out[k].replace(HELPERS.delay + '\n\n\n', '').replace('\n\n\n' + HELPERS.delay, '').replace(HELPERS.delay, '');
      if (!out[k].trim()) out.splice(k, 3);
      const shift = before - out.join('\n').split('\n').length;
      for (const [id, ls] of lineMap) lineMap.set(id, ls.map(l => l - shift));
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
    if (n.type === 'reshape') return 'Reshape';
    if (['heads', 'merge', 'transpose', 'scale', 'mask', 'rope'].includes(n.type)) return { heads: 'Split heads', merge: 'Merge heads', transpose: 'Transpose', scale: 'Scale', mask: 'Causal mask', rope: 'Rotary embedding' }[n.type];
    if (n.type === 'conv') return `${p.transposed ? 'ConvTranspose' : 'Conv'} ${p.k}×${p.k} ${p.in}→${p.out}`;
    if (n.type === 'pool') return p.kind === 'adaptive' ? 'Adaptive avg pool' : (p.kind === 'avg' ? 'Avg pool ' : 'Max pool ') + p.k;
    if (n.type === 'upsample') return 'Upsample ×' + p.scale;
    if (n.type === 'delay') return 'Delay ' + (p.name || '');
    if (n.type === 'token') return 'Class token';
    if (n.type === 'param') return 'Parameter ' + (p.name || '');
    if (n.type === 'conv1d') return 'Causal conv';
    if (n.type === 'noise') return 'Noise';
    if (n.type === 'select') return `Position ${p.index}`;
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
          while (!isOp(']')) { if (isOp(':')) { next(); idx.push(isOp(',') || isOp(']') ? { k: 'slice', full: true } : { k: 'slice' }); } else { const x = expr(); if (isOp(':')) { next(); if (!isOp(']') && !isOp(',')) expr(); idx.push({ k: 'slice' }); } else idx.push(x); } if (isOp(',')) next(); }
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
      case 'bin': {
        // brackets where precedence needs them: (d / n_head) ** -0.5, a - (b - c)
        const P = { '**': 4, '*': 3, '/': 3, '//': 3, '%': 3, '@': 3, '+': 2, '-': 2 }, me = P[e.op] || 1;
        const side = (x, right) => { const t = u(x), q = x && x.k === 'bin' ? P[x.op] || 1 : 9; return q < me || (q === me && (e.op === '**' ? !right : right)) ? '(' + t + ')' : t; };
        return side(e.l, false) + ' ' + e.op + ' ' + side(e.r, true);
      }
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
    const symCount = {}, followerMods = [], madeGroups = [];
    let inFwd = false; // a parameter read in forward() is a node; in __init__ it is the parameter itself
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
      // an operand in brackets when it is itself a sum, or any compound under * / **; a negative exponent stays bare
      const s = (v, right) => { const t = valStr(v); if (op === '**' && right && typeof v === 'number') return t; return /[\s+\-]/.test(t) || ((op === '**' || ((op === '*' || op === '/' || op === '//') && right)) && /[*/]/.test(t)) ? '(' + t + ')' : t; };
      if ((a && a.sym) || (b && b.sym)) return { sym: `${s(a, false)}${op === '//' ? '//' : op}${s(b, true)}`.replace(/\*\*/, '**') };
      return null;
    }

    function ev(e, env) {
      switch (e.k) {
        case 'const': return e.v;
        case 'str': return { str: e.v };
        case 'name': if (env.has(e.v)) return env.get(e.v); return { path: e.v };
        case 'attr': {
          const base = ev(e.v, env);
          if (base && !isT(base) && (base.mod || base.list || (base.inst && base !== env.get('self'))) && e.name === 'requires_grad_') return { freezeOf: base };
          // self.target.load_state_dict(self.encoder.state_dict()): target follows encoder (an EMA tie)
          if (base && !isT(base) && (base.mod || (base.inst && base !== env.get('self'))) && e.name === 'load_state_dict') return { loadInto: base };
          if (base && !isT(base) && (base.mod || (base.inst && base !== env.get('self'))) && e.name === 'state_dict') return { stateOf: base };
          if (base && base.inst) {
            if (!(e.name in base.inst.attrs)) return { path: 'self.' + e.name, missing: true };
            const v = base.inst.attrs[e.name];
            // a learned tensor read in forward is a Parameter node (a class token's .expand turns it back, below)
            if (v && v.mod === 'param' && inFwd) {
              // one node per parameter for each call (a table read twice is the same table)
              if (v._pnode && v._pgroup === (curGroup && curGroup.id) && g.nodes.some(z => z.id === v._pnode)) { const r0 = { t: v._pnode }; r0.paramMod = v; return r0; }
              const r = addNode('param', Object.assign({ shape: v.shape.map(dimStr).join(', '), init: v.init || 'zeros', init_arg: v.init_arg || '', name: v.attrName || e.name }, v.frozen ? { frozen: true } : {}), []); r.paramMod = v; v._pnode = r.t; v._pgroup = curGroup && curGroup.id; return r; }
            return v;
          }
          if (isT(base) && base.paramMod && (e.name === 'expand' || e.name === 'requires_grad_')) {
            g.nodes = g.nodes.filter(z => z.id !== base.t); for (const gr of g.groups) gr.nodes = gr.nodes.filter(id => id !== base.t); if (curGroup) curGroup.nodes = curGroup.nodes.filter(id => id !== base.t);
            base.paramMod._pnode = null;
            return e.name === 'expand' ? { paramExpandOf: base.paramMod } : { freezeOf: base.paramMod };
          }
          if (isT(base)) return { method: e.name, self: base };
          if (base && base.list && e.name === 'append') return { listAppend: base };
          if (base && base.mod && (e.name === 'weight' || e.name === 'bias')) return { param: e.name, of: base };
          if (base && base.mod === 'param' && e.name === 'expand') return { paramExpandOf: base };
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
            const cs = v => String(valStr(v)).replace(/\s*\*\*\s*/g, ' ** '); // d**0.5 written as d ** 0.5
            // a power written out (512 ** 0.5) stays written out rather than becoming 22.627…
            // a constant written as arithmetic of sizes stays as written, with the sizes' self. dropped
            if (isT(a)) return addNode('binop', { op: op || '+', const: e.r.k === 'bin' && (typeof b === 'number' || (b && b.sym)) ? unparse(e.r, env).replace(/\bself\./g, '') : cs(b) }, [a]);
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
          if (isT(v) && e.idx.length === 2 && e.idx[0].k === 'slice' && (ev(e.idx[1], env) || {}).timeIdx) return v; // x[:, t] in a time loop
          if (isT(v) && e.idx.length === 2 && e.idx[0].full && typeof ev(e.idx[1], env) === 'number') return addNode('select', { index: String(ev(e.idx[1], env)) }, [v]); // x[:, 0]
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
      'F.silu': 'silu', 'F.relu': 'relu', 'F.gelu': 'gelu', 'F.sigmoid': 'sigmoid', 'F.tanh': 'tanh', 'F.softmax': 'softmax', 'F.softplus': 'softplus',
      'torch.sigmoid': 'sigmoid', 'torch.tanh': 'tanh', 'torch.relu': 'relu', 'torch.softmax': 'softmax',
      'torch.nn.functional.silu': 'silu', 'torch.nn.functional.relu': 'relu', 'torch.nn.functional.gelu': 'gelu', 'torch.nn.functional.softmax': 'softmax',
    };
    const normPath = p => p.replace(/^torch\.nn\.functional\./, 'F.').replace(/^torch\.nn\./, 'nn.');

    // a rearrangement call (unflatten, flatten, transpose, permute, squeeze, unsqueeze) becomes a Reshape node; one
    // applied straight to another's result (x.unflatten(…).transpose(1, 2)) joins it, so a chain is one node
    function reshapeCall(step, self, chained) {
      const prev = chained && self && g.nodes.find(z => z.id === self.t);
      if (prev && prev.type === 'reshape' && Array.isArray(prev.params._steps) && !g.edges.some(z => z.from === prev.id)) { prev.params._steps.push(step); return { t: prev.id }; }
      return addNode('reshape', { pattern: '', sizes: '', _steps: [step] }, [self]);
    }
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
        if (p === 'split_heads' && isT(A[0])) return addNode('reshape', { pattern: '', sizes: '', _steps: [{ op: 'unflatten', dim: -1, sizes: [dimStr(A[1]), '-1'] }, { op: 'transpose', a: 1, b: 2 }] }, [A[0]]);
        if (p === 'merge_heads' && isT(A[0])) return addNode('reshape', { pattern: '', sizes: '', _steps: [{ op: 'transpose', a: 1, b: 2 }, { op: 'flatten', a: 2, b: null }] }, [A[0]]);
        if (p === 'causal_mask' && isT(A[0])) return addNode('mask', {}, [A[0]]);
        if (p === 'dt_bias_' && A[0] && A[0].param === 'bias' && A[0].of.mod === 'linear') { A[0].of.bias_init = 'dt'; return null; }
        if (p === 'causal_conv' && A[0] && A[0].mod === 'conv1d' && isT(A[1])) { const m = A[0]; return addNode('conv1d', Object.assign({ ch: dimStr(m.ch), k: dimStr(m.k), bias: m.bias !== false, name: m.attrName || '' }, m.frozen ? { frozen: true } : {}), [A[1]]); }
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
          if (target && target.mod === 'param' && fn === 'normal') { target.init = 'normal'; target.init_arg = valStr('std' in KW ? KW.std : A.length > 2 ? A[2] : 1); return null; }
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
        // images
        if (p === 'nn.ConvTranspose2d') return { mod: 'conv', transposed: true, in: arg(0, 'in_channels'), out: arg(1, 'out_channels'), k: arg(2, 'kernel_size'), stride: arg(3, 'stride', 1), pad: arg(4, 'padding', 0), bias: arg(7, 'bias', true) };
        if (p === 'nn.Conv2d') return { mod: 'conv', in: arg(0, 'in_channels'), out: arg(1, 'out_channels'), k: arg(2, 'kernel_size'), stride: arg(3, 'stride', 1), pad: arg(4, 'padding', 0), bias: arg(7, 'bias', true) };
        if (p === 'nn.BatchNorm2d') return { mod: 'norm', kind: 'BatchNorm2d', dim: arg(0, 'num_features'), eps: KW.eps };
        if (p === 'nn.MaxPool2d' || p === 'nn.AvgPool2d') return { mod: 'pool', kind: p === 'nn.MaxPool2d' ? 'max' : 'avg', k: arg(0, 'kernel_size'), stride: arg(1, 'stride', null), pad: arg(2, 'padding', 0) };
        if (p === 'nn.AdaptiveAvgPool2d') return { mod: 'pool', kind: 'adaptive', size: arg(0, 'output_size', 1) };
        if (p === 'nn.Upsample') return { mod: 'upsample', scale: arg(1, 'scale_factor', 2), mode: arg(2, 'mode', { str: 'nearest' }) };
        if (p === 'nn.LeakyReLU') return { mod: 'act', fn: 'leaky_relu', slope: arg(0, 'negative_slope', 0.01), dim: -1 };
        if (p === 'nn.Flatten') return { mod: 'flatten', start: arg(0, 'start_dim', 1) };
        if (p === 'nn.Parameter' && A[0] && A[0].zerosShape) return { mod: 'param', shape: A[0].zerosShape };
        if (p === 'nn.Parameter' && A[0] && A[0].onesShape) return { mod: 'param', shape: A[0].onesShape, init: 'ones' };
        if (p === 'nn.Parameter' && e.args[0]) {
          let z0 = e.args[0], shp = null;
          if (z0.k === 'call' && z0.fn.k === 'attr' && z0.fn.name === 'reshape') { shp = z0.args.map(a => ev(a, env)); z0 = z0.fn.v; }
          if (z0.k === 'call' && unparse(z0.fn, env) === 'sincos_2d' && z0.args.length === 2) {
            const nn_ = ev(z0.args[0], env), dd = ev(z0.args[1], env);
            for (const w of ['Ignored call to sincos_2d().', 'Ignored call to ?.reshape().']) { const i = warnings.lastIndexOf(w); if (i >= 0) warnings.splice(i, 1); }
            return { mod: 'param', shape: shp || [nn_, dd], init: 'sincos2d' };
          }
          // torch.log(torch.arange(1, N + 1, dtype=torch.float32)).repeat(rows, 1): Mamba's S4D-real init of A_log
          let z = e.args[0], rows = [];
          if (z.k === 'call' && z.fn.k === 'attr' && z.fn.name === 'repeat') { rows = z.args.slice(0, -1); z = z.fn.v; }
          const isCall = (x, nm) => x && x.k === 'call' && unparse(x.fn, env) === nm;
          if (isCall(z, 'torch.log') && isCall(z.args[0], 'torch.arange') && z.args[0].args.length === 2) {
            const top = ev(z.args[0].args[1], env), N = typeof top === 'number' ? top - 1 : top && top.sym ? { sym: top.sym.replace(/^\((.*)\)$/, '$1').replace(/\s*\+\s*1$/, '').replace(/^\((.*)\)$/, '$1') } : null;
            if (N != null) {
              for (const w of ['Ignored call to torch.arange().', 'Ignored call to torch.log().', 'Ignored call to ?.repeat().']) { const i = warnings.lastIndexOf(w); if (i >= 0) warnings.splice(i, 1); }
              return { mod: 'param', shape: rows.map(r => ev(r, env)).concat(N), init: 's4d' };
            }
          }
        }
        if (p === 'nn.Conv1d') return { mod: 'conv1d', ch: arg(0, 'in_channels'), k: arg(2, 'kernel_size'), bias: arg(7, 'bias', true) };
        if (p === 'nn.Linear') return { mod: 'linear', in: arg(0, 'in_features'), out: arg(1, 'out_features'), bias: arg(2, 'bias', true) };
        if (p === 'nn.LayerNorm' || p === 'nn.RMSNorm') return { mod: 'norm', kind: p.slice(3), dim: arg(0, 'normalized_shape'), eps: KW.eps, affine: KW.elementwise_affine === false ? false : undefined };
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
        if ((p === 'F.max_pool2d' || p === 'F.avg_pool2d') && isT(A[0])) { const st = arg(2, 'stride', null); return addNode('pool', { kind: p === 'F.max_pool2d' ? 'max' : 'avg', k: valStr(arg(1, 'kernel_size', 2)), stride: st == null ? '' : valStr(st), pad: valStr(arg(3, 'padding', 0)) }, [A[0]]); }
        if (p === 'F.adaptive_avg_pool2d' && isT(A[0])) return addNode('pool', { kind: 'adaptive', size: valStr(arg(1, 'output_size', 1)) }, [A[0]]);
        if (p === 'F.interpolate' && isT(A[0])) { const md = arg(3, 'mode', null); return addNode('upsample', { scale: valStr(arg(2, 'scale_factor', 2)), mode: md && md.str ? md.str : 'nearest' }, [A[0]]); }
        if (p === 'F.leaky_relu' && isT(A[0])) return addNode('act', { fn: 'leaky_relu', slope: valStr(arg(1, 'negative_slope', 0.01)), dim: '-1' }, [A[0]]);
        if (p === 'torch.flatten' && isT(A[0])) return addNode('func', { fn: '.flatten', extra: valStr(arg(1, 'start_dim', 0)), n: 1 }, [A[0]]);
        if (FN_ACT[p] && tens.length) return applyAct(FN_ACT[p] === 'gelu' && KW.approximate && KW.approximate.str === 'tanh' ? 'gelu_tanh' : FN_ACT[p], A[0], arg(1, 'dim', -1));
        if ((p === 'F.dropout' || p === 'nn.functional.dropout') && tens.length) return A[0];
        if ((p === 'torch.add' || p === 'torch.mul') && tens.length === 2) return addNode('binop', { op: p === 'torch.add' ? '+' : '*' }, tens);
        if (p === 'torch.randn_like' && tens.length === 1 && !Object.keys(KW).length) return addNode('noise', {}, [tens[0]]);
        if (p === 'torch.zeros' && !tens.length && A.length >= 1 && !Object.keys(KW).length) return { zeros: true, zerosShape: A };
        if (p === 'torch.ones' && !tens.length && A.length >= 1 && !Object.keys(KW).length) return { ones: true, onesShape: A };
        if ((p === 'torch.zeros' || p === 'torch.zeros_like') && !tens.length) return { zeros: true };
        if (p === 'torch.zeros_like' && tens.length) return { zeros: true };
        // the steps a time loop collected, stacked back along time: the per-step value is the sequence
        if (p === 'torch.stack' && A[0] && A[0].steps && A[0].list.length === 1 && isT(A[0].list[0]) && +valStr(arg(1, 'dim', 0)) === 1) return A[0].list[0];
        // torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1): learned tokens in front of the sequence
        if (p === 'torch.cat' && A[0] && A[0].list && A[0].list.length === 2 && A[0].list[0] && A[0].list[0].paramExpand && isT(A[0].list[1]) && +valStr(arg(1, 'dim', 0)) === 1) {
          const pm = A[0].list[0].paramExpand, sh = pm.shape;
          return addNode('token', Object.assign({ n: dimStr(sh[sh.length - 2]), dim: dimStr(sh[sh.length - 1]), name: pm.attrName || '' }, pm.frozen ? { frozen: true } : {}), [A[0].list[1]]);
        }
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
        if (m === 'new_zeros' || m === 'zeros_like') return { zeros: true };
        const chained = e.fn && e.fn.k === 'attr' && e.fn.v && e.fn.v.k === 'call', I = v => { const x = num2(v); return typeof x === 'number' ? x : null; };
        const ints = A.every(v => I(v) != null) && !Object.keys(KW).length;
        if (m === 'flatten' && ints && A.length <= 2) return reshapeCall({ op: 'flatten', a: A.length ? I(A[0]) : 0, b: A.length > 1 ? I(A[1]) : null }, self, chained);
        if (m === 'unflatten' && A.length === 2 && I(A[0]) != null && A[1] && A[1].list) return reshapeCall({ op: 'unflatten', dim: I(A[0]), sizes: A[1].list.map(v => (I(v) === -1 ? '-1' : dimStr(v))) }, self, chained);
        if ((m === 'unsqueeze' || m === 'squeeze') && A.length === 1 && ints) return reshapeCall({ op: m, dim: I(A[0]) }, self, chained);
        if (m === 'permute' && ints && A.length > 1) return reshapeCall({ op: 'permute', perm: A.map(I) }, self, chained);
        if (m === 'permute' && A.length === 1 && A[0] && A[0].list) return reshapeCall({ op: 'permute', perm: A[0].list.map(I) }, self, chained);
        if (m === 'dim') return null;
        if (m === 'transpose' && ['-2,-1', '-1,-2'].includes(A.map(x => valStr(x)).join(','))) return reshapeCall({ op: 'transpose', a: -2, b: -1 }, self, chained);
        // norm(x.transpose(1, 2)).transpose(1, 2): the channel-first idiom collapses back into one norm layer
        const TR = new Set(['1, 2', '2, 1', '1, -1', '-1, 1']);
        if (m === 'transpose' && TR.has(e.args.map(a => unparse(a, env)).join(', '))) {
          const nn_ = g.nodes.find(z => z.id === self.t);
          const inE = nn_ && nn_.type === 'norm' && NORMS[nn_.params.kind] && NORMS[nn_.params.kind].channelFirst && g.edges.find(z => z.to === nn_.id);
          const tr = inE && g.nodes.find(z => z.id === inE.from);
          const trS = tr && tr.type === 'reshape' && tr.params._steps && tr.params._steps.length === 1 && tr.params._steps[0].op === 'transpose' ? tr.params._steps[0] : null;
          if (tr && trS && TR.has(`${trS.a}, ${trS.b}`) && g.edges.filter(z => z.from === tr.id).length === 1) {
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
        if (m === 'transpose' && ints && A.length === 2) return reshapeCall({ op: 'transpose', a: I(A[0]), b: I(A[1]) }, self, chained);
        const extra = e.args.map(a => unparse(a, env)).concat(Object.entries(e.kw).map(([a, b]) => a + '=' + unparse(b, env)));
        const tens = A.filter(isT);
        warnings.push(`Tensor method .${m}() is kept as a custom op.`);
        return addNode('func', { fn: '.' + m, extra: extra.join(', '), n: 1 + tens.length }, [self, ...tens]);
      }
      if (fv && fv.paramExpandOf) return { paramExpand: fv.paramExpandOf };
      if (fv && fv.stateOf) return { stateDictOf: fv.stateOf };
      if (fv && fv.loadInto) { if (A[0] && A[0].stateDictOf) { fv.loadInto.followsMod = A[0].stateDictOf; followerMods.push(fv.loadInto); } return null; }
      // self.fc.requires_grad_(False): frozen (a ModuleList: every block in it)
      if (fv && fv.freezeOf) { if (A.length && A[0] === false) { const v = fv.freezeOf; (v.list ? v.list : [v]).forEach(x => { if (x) x.frozen = true; }); } return null; }
      if (fv && fv.listAppend) { fv.listAppend.list.push(A[0]); fv.listAppend.steps = true; return null; }
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
    // a frozen module's layer (self.fc.requires_grad_(False) in __init__) is marked frozen
    function applyMod(m, args, attrName) {
      const r = applyMod0(m, args, attrName);
      if (m && !m.inst && r && r.t != null && m._nodeId == null) m._nodeId = r.t;
      if (m && m.frozen && !m.inst && r && r.t != null) { const nd = g.nodes.find(z => z.id === r.t); if (nd && FREEZABLE.has(nd.type)) nd.params.frozen = true; }
      return r;
    }
    function applyMod0(m, args, attrName) {
      const x = args[0];
      if (m.inst) return callInstance(m, args, attrName);
      switch (m.mod) {
        case 'embed': {
          const ei = { init: m.init || 'default', init_arg: m.init_arg || '' };
          const r = x && x.arange ? addNode('embed', Object.assign({ num: dimStr(m.num), dim: dimStr(m.dim), positions: true, name: m.attrName || '' }, ei), [x.arange])
            : addNode('embed', Object.assign({ num: dimStr(m.num), dim: dimStr(m.dim), positions: false, name: m.attrName || '' }, ei), [x]);
          m._node = r.t; return r;
        }
        case 'linear': { const r = addNode('linear', { in: dimStr(m.in), out: dimStr(m.out), bias: m.bias !== false, init: m.init || 'default', init_arg: m.init_arg || '', zero_bias: !!m.zero_bias, bias_init: m.bias_init || '', name: m.attrName || '' }, [x]); m._node = r.t; return r; }
        case 'norm': return addNode('norm', Object.assign({ kind: m.kind, dim: dimStr(m.dim), groups: num2(m.groups) || 8, eps: m.eps != null ? valStr(m.eps) : '', name: m.attrName || '' }, m.affine === false ? { affine: false } : {}), [x]);
        case 'dropout': return addNode('dropout', { p: m.p, name: m.attrName || '' }, [x]);
        case 'act': return m.fn === 'leaky_relu' && isT(x) ? addNode('act', { fn: 'leaky_relu', slope: valStr(m.slope), dim: '-1' }, [x]) : applyAct(m.fn, x, m.dim);
        case 'conv': return addNode('conv', { in: dimStr(m.in), out: dimStr(m.out), k: dimStr(m.k), stride: dimStr(m.stride), pad: dimStr(m.pad), bias: m.bias !== false, transposed: !!m.transposed, name: m.attrName || '' }, [x]);
        case 'pool': return addNode('pool', { kind: m.kind, k: m.k != null ? valStr(m.k) : '2', stride: m.stride == null ? '' : valStr(m.stride), pad: valStr(m.pad ?? 0), size: valStr(m.size ?? 1) }, [x]);
        case 'upsample': return addNode('upsample', { scale: valStr(m.scale), mode: m.mode && m.mode.str ? m.mode.str : 'nearest' }, [x]);
        case 'flatten': return addNode('func', { fn: '.flatten', extra: valStr(m.start), n: 1 }, [x]);
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
            const lit = v;
            v = newSym(p.name, v);
            // a number written inside another class's __init__ (DarknetConv(c, c // 2, 1, 1)) is a constant of that
            // class: a size of the class it creates (so that one keeps its arguments), but not of the class itself
            const ci = callerEnv.get && callerEnv.get('self') && callerEnv.get('self').inst;
            if (ci && ci !== rootInst) (ci.consts = ci.consts || {})[v.sym] = lit;
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
        curGroup = { id: uid('g'), frozen: !!m.frozen && !m.followsMod, parent: parent ? parent.id : null, cls: m.inst.cls, attr: pyIdent(attrName || m.inst.cls.toLowerCase() + nth), label: `${m.inst.cls.replace(/([a-z])([A-Z])/g, '$1 $2')} ${nth}`, nodes: [], argHints: Object.assign({}, m.inst.hints), argDefaults: Object.assign({}, m.inst.defaults), consts: Object.assign({}, m.inst.consts) };
        g.groups.push(curGroup); if (m._groupId == null) m._groupId = curGroup.id; madeGroups.push(curGroup);
        if (m.followsMod) curGroup.followsMod = m.followsMod;
      }
      const env = new Map([['self', m]]);
      fwd.params.slice(1).forEach((p, i) => env.set(p.name, i < args.length ? args[i] : p.def ? ev(p.def, env) : null));
      const wasFwd = inFwd; inFwd = true;
      const r = exec(fwd.body, env);
      inFwd = wasFwd;
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
    // ── a loop over time: for t in range(x.size(1)): … read as one step, with Delays for what it carries ──
    function isTimeLoop(s, env) {
      const it = s.iter;
      if (s.target.k !== 'name' || it.k !== 'call' || pathOf(it.fn) !== 'range' || it.args.length !== 1) return false;
      const a = it.args[0];
      const sz = a.k === 'call' && a.fn.k === 'attr' && a.fn.name === 'size' && a.args.length === 1 && a.args[0].k === 'const' && a.args[0].v === 1;
      const sh = a.k === 'sub' && a.v.k === 'attr' && a.v.name === 'shape' && a.idx.length === 1 && a.idx[0].k === 'const' && a.idx[0].v === 1;
      return (sz || sh) && isT(ev(sz ? a.fn.v : a.v.v, env));
    }
    const namesIn = (e, out = new Set()) => { if (!e || typeof e !== 'object') return out; if (e.k === 'name') out.add(e.v); for (const v of Object.values(e)) if (Array.isArray(v)) v.forEach(x => namesIn(x, out)); else if (v && typeof v === 'object') namesIn(v, out); return out; };
    function timeLoop(s, env) {
      // the variables it carries from step to step: set before the loop, read in it before they are set again
      const carried = [], set = new Set();
      for (const st of s.body) {
        if (st.k === 'assign') { namesIn(st.value).forEach(nm => { if (!set.has(nm) && env.has(nm) && !carried.includes(nm) && s.body.some(b => b.k === 'assign' && b.targets.some(t => t.k === 'name' && t.v === nm))) carried.push(nm); }); st.targets.forEach(t => { if (t.k === 'name') set.add(t.v); }); }
        else namesIn(st).forEach(nm => { if (!set.has(nm) && env.has(nm) && !carried.includes(nm) && s.body.some(b => b.k === 'assign' && b.targets.some(t => t.k === 'name' && t.v === nm))) carried.push(nm); });
      }
      const delays = carried.map(nm => {
        const v0 = env.get(nm);
        if (v0 && !v0.zeros) warnings.push(`The loop's starting ${nm} isn't zeros; the Delay starts from zeros.`);
        const d = addNode('delay', { name: nm }, []); env.set(nm, d); return [nm, d];
      });
      env.set(s.target.v, { timeIdx: true });
      exec(s.body, env);
      for (const [nm, d] of delays) { const v = env.get(nm); if (isT(v)) g.edges.push({ id: uid('e'), from: v.t, to: d.t, port: 0 }); else warnings.push(`${nm} isn't a tensor at the end of the loop.`); }
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
            if (isTimeLoop(s, env)) { timeLoop(s, env); break; }
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
    inFwd = true;
    let ret = exec(fwd.body, fenv);
    inFwd = false;
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
        const mk = /^(?:([\d.eE+-]+)\s*\*\s*)?kl_normal\((\w+),\s*(\w+)\)$/.exec(term);
        if (mk) {
          const o1 = outs.find(z => z.params.name === mk[2]), o2 = outs.find(z => z.params.name === mk[3]);
          if (!o1 || !o2) { warnings.push(`loss(): kl_normal needs two outputs of forward().`); continue; }
          addNode('loss', { fn: 'kl', weight: mk[1] ? String(+mk[1]) : '1' }, [{ t: g.edges.find(e => e.to === o1.id).from }, { t: g.edges.find(e => e.to === o2.id).from }]);
          continue;
        }
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
        const tOut = outs.find(z => z.params.name === tname && z !== o);
        if (tOut) { addNode('loss', p, [{ t: g.edges.find(e => e.to === o.id).from }, { t: g.edges.find(e => e.to === tOut.id).from }]); continue; }
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
      for (const n of g.nodes) for (const f of ['in', 'out', 'dim', 'init', 'num', 'heads', 'k', 'stride', 'pad', 'ch', 'shape', 'n', 'sizes']) if (typeof n.params[f] === 'string') n.params[f] = f === 'sizes' ? mapSizes(n.params[f], v => v.replace(IDENT, t => ren.get(t) || t)) : n.params[f].replace(IDENT, t => ren.get(t) || t);
      for (const n of g.nodes) if (n.params._steps) n.params._steps.forEach(st => { if (st.sizes) st.sizes = st.sizes.map(v => String(v).replace(IDENT, t => ren.get(t) || t)); });
      // block uses' sizes and the module calls' name hints follow the renamed sizes
      for (const n of g.nodes) if (n.type === 'block' && n.params.args) for (const k of Object.keys(n.params.args)) n.params.args[k] = String(n.params.args[k]).replace(IDENT, t => ren.get(t) || t);
      for (const gr of g.groups || []) if (gr.argHints) gr.argHints = Object.fromEntries(Object.entries(gr.argHints).map(([k, v]) => [ren.get(k) || k, v]));
    }
    // infer input shapes from first consumer
    const ix = index(g);
    const images = [];
    for (const n of g.nodes.filter(n => n.type === 'input')) {
      const seen = new Set(); let dim = null, img = null;
      const walk = id => { if (dim || img || seen.has(id)) return; seen.add(id); for (const o of ix.outs.get(id)) { const c = ix.byId.get(o.to); if (c.type === 'conv') { img = c.params.in; return; } if (c.type === 'linear') { dim = c.params.in; return; } if (c.type === 'norm') { dim = c.params.dim; return; } walk(c.id); } };
      const feedsEmbed = ix.outs.get(n.id).some(o => ix.byId.get(o.to).type === 'embed');
      if (feedsEmbed) { n.params.shape = 'e, t'; if (g.symbols.t == null) g.symbols.t = 64; continue; }
      walk(n.id);
      // an image (it reaches a convolution first): e × C × H × W, the size worked out below
      if (img) { n.params.shape = `e, ${img}, 224, 224`; images.push(n); continue; }
      n.params.shape = (seqInput ? 'e, t, ' : 'e, ') + (dim || 'd_in');
      if (seqInput && g.symbols.t == null) g.symbols.t = 16;
    }
    // the __main__ test call, when there is one, gives each input's real shape: model(torch.randn(2, 1024, 1176), …)
    {
      const m = /\n\s*\w+(?:\s*,\s*\w+)*\s*=\s*model\(([^\n]*)\)\s*\n/.exec(src.slice(src.indexOf('__main__')));
      const split = t => { const out = []; let dep = 0, cur = ''; for (const ch of t) { if ('([{'.includes(ch)) dep++; if (')]}'.includes(ch)) dep--; if (ch === ',' && !dep) { out.push(cur.trim()); cur = ''; } else cur += ch; } if (cur.trim()) out.push(cur.trim()); return out; };
      const dimsOf = a => { let r = /^torch\.(?:randn|rand|zeros|ones|empty)\((.*)\)$/.exec(a); if (r) return split(r[1]).filter(x => !x.includes('=')).map(Number);
        r = /^torch\.randint\((.*)\)$/.exec(a); if (r) { const p = split(r[1]), tup = p.find(x => x.startsWith('(')); return tup ? split(tup.slice(1, -1)).map(Number) : null; } return null; };
      const ins = g.nodes.filter(n => n.type === 'input' && !newTargets.some(x => x.n === n)), args = m ? split(m[1]) : [];
      // per-input example lines: name = expression  # (n=512, c_in=1176) — the shape with its size names, and the example
      const main = src.slice(src.indexOf('__main__')), byName = new Map(ins.map(n => [pyIdent(n.params.name || 'x'), n]));
      for (const mm of main.matchAll(/^ {4}(\w+) = (.+?)  # \((.*)\)$/gm)) {
        const n = byName.get(mm[1]); if (!n) continue;
        const dims = mm[3].split(',').map(x => x.trim()).filter(Boolean).map(d => { const kv = /^([A-Za-z_]\w*)=(-?\d+)$/.exec(d); if (kv) { g.symbols[kv[1]] = +kv[2]; return kv[1]; } return d; });
        n.params.shape = dims.join(', '); n.params.demoRead = true;
        if (!/^torch\.(randn|randint)\([\d\s,()]*\)$|^torch\.rand\(\d+, \d+\)\.argsort\(dim=1\)\[:, :\d+\]$/.test(mm[2])) n.params.demo = mm[2]; // a plain example is written again from the shape
      }
      const read = ins.filter(n => n.params.demoRead);
      // model(x) with x = torch.randn(…) on a line of its own
      const argOf = a => { if (!/^[A-Za-z_]\w*$/.test(a || '')) return a; const am = new RegExp('^\\s*' + a + '\\s*=\\s*(torch\\.\\w+\\(.*\\))\\s*$', 'm').exec(main); return am ? am[1] : a; };
      // an image's size: from the example call when there is one, otherwise the size that gives the first Linear
      // after the convolutions the width it was written with (64 * 7 * 7 → 28 × 28), otherwise 224 × 224
      for (const n of images) {
        if (n.params.demoRead) continue;
        const i = ins.indexOf(n);
        const dims = m && args.length === ins.length ? dimsOf(argOf(args[i])) : null;
        const C = parseShape(n.params.shape)[1];
        if (dims && dims.length === 4 && dims.every(Number.isInteger)) { n.params.shape = `e, ${C}, ${dims[2]}, ${dims[3]}`; if (g.symbols[BATCH] == null || g.symbols[BATCH] === 8) g.symbols[BATCH] = dims[0]; n.params.imgRead = true; continue; }
        const lin = g.nodes.filter(z => z.type === 'linear');
        if (!lin.length) continue;
        // (when several sizes fit, e.g. after global average pooling, 224 × 224)
        const fits = [224, 28, 32, 64, 96, 128, 256, 299, 416, 512, 16, 48, 112, 384, 448, 640].filter(hw => {
          n.params.shape = `e, ${C}, ${hw}, ${hw}`;
          const sh = inferShapes(g).shapes, ix2 = index(g);
          return lin.every(z => { const s0 = ix2.inKeys.get(z.id)[0], x = s0 && sh.get(s0); return x && dimEq(x[x.length - 1], z.params.in, g.symbols) === true; });
        });
        n.params.shape = `e, ${C}, ${fits.length ? fits[0] : 224}, ${fits.length ? fits[0] : 224}`;
      }
      if (m && args.length === ins.length) ins.forEach((n, i) => {
        if (n.params.demoRead || images.includes(n)) { delete n.params.imgRead; return; }
        const dims = dimsOf(argOf(args[i])); if (!dims || dims.some(x => !Number.isInteger(x))) return;
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
    // followers (EMA ties): a layer follows the layer made from the module it copies; a block use, the other use
    for (const m of followerMods) {
      if (m.inst) continue;
      const n = m._nodeId != null && g.nodes.find(z => z.id === m._nodeId), lead = m.followsMod._nodeId;
      if (n && lead != null) { n.params.follows = lead; delete n.params.frozen; } else warnings.push('A load_state_dict copy refers to a layer forward() never uses; it was left out.');
    }
    const emaM = findMethod(src, root.name, 'ema_update'), moms = new Map();
    if (emaM) for (const mm of emaM.body.join('\n').matchAll(/m = ([\d.eE+-]+)(?: \+ \(([\d.eE+-]+) - [\d.eE+-]+\) \* progress)?\s*# (\w+) follows (\w+)/g)) moms.set(mm[3], [mm[1], mm[2] || mm[1]]);
    const setMom = n => { const v = moms.get(n.params.name); if (v) { n.params.momentum = v[0]; n.params.momentum_end = v[1]; } else { n.params.momentum = '0.996'; n.params.momentum_end = '1.0'; } };
    for (const n of g.nodes) if (n.params.follows) setMom(n);
    // reshape chains read from code become patterns, named after the sizes they act on
    if (g.nodes.some(n => n.type === 'reshape' && n.params._steps)) {
      const { shapes } = inferShapes(g), ix2 = index(g);
      for (const n of g.nodes) if (n.type === 'reshape' && n.params._steps) {
        const x = shapes.get((ix2.ins.get(n.id) || [])[0]); const r = x ? stepsToPattern(n.params._steps, x) : null;
        if (r) { n.params.pattern = r.pattern; n.params.sizes = r.sizes; delete n.params._steps; }
      }
    }
    linkSizes(g);
    groupsToBlocks(g);
    for (const gr of madeGroups) if (gr.followsMod && gr.followsMod._groupId != null) {
      const lead = madeGroups.find(z => z.id === gr.followsMod._groupId), use = g.nodes.find(z => z.id === gr.useId), luse = lead && g.nodes.find(z => z.id === lead.useId);
      if (use && luse) { use.params.follows = luse.id; delete use.params.frozen; setMom(use); } else warnings.push('A block that follows another (load_state_dict) could not be matched to its use; the tie was left out.');
    }
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
  function mergeIntoSource(src, gen, used) {
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

  return { lowerReshapes, reshapeSteps, reshapeStepCode, reshapeShape, reshapeAxisSource, stepsToPattern, reshapeDescribe, reshapePlan, stepsShape, parseSizes, HELPER_OPS, indexInputRange, FREEZABLE, helperDoc, srcKey, splitKey, blockOutputs, blockClassCode, LOSSES, lossParts, lossPlan, weightShape, tieOwner, blockParams, blockInputs, expandBlocks, instantiateBlock, presetResidualMLPBlocks, NORMS, linkSizes, syncWidths, mergeIntoSource, uid, BATCH, DEFAULTS, ACTS, INITS, initCode, makeNode, inCount, isCircle, index, topo, evalDim, parseShape, inferShapes, paramCount, propagateWidth, generate, importPython, importFx, autoLayout, nodeSize, nodeLabel, presetResidualMLP, parse, identsIn };
})();
if (typeof module !== 'undefined') module.exports = NN;
