// ───────────────────────── Torch Canvas: expected variance at init ─────────────────────────
// Predicts, without building or running the network, the variance of the activations (forward
// pass of a standard Gaussian input) and of the gradients (backward pass of a standard Gaussian
// gradient at the outputs) at every Linear layer right after initialization.
//
// Mean-field rules. A tensor is carried as a (positions × samples) array: each sample column is
// one feature coordinate in one random "world", and its values at different positions are
// correlated the way the real network's are. Elementwise ops (activations, gates, residual adds)
// and norms are applied exactly to these values. Weights are never drawn: they are averaged out,
// so a Linear's output is a fresh Gaussian, independent of everything before it, whose
// covariance between positions t and u is
//     fan_in · Var(w) · E[x_t x_u] + Var(b)                 (forward)
//     fan_out · Var(w) · E[dy_t dy_u]                       (backward, gradient independence)
// Attention is computed on the arrays themselves: groups of sample columns act as heads, so the
// components that positions share, and the way softmax cancels them, come out exactly.
// Runs in the browser and in Node.
const NNVar = (() => {
  'use strict';
  const C = typeof NN !== 'undefined' ? NN : require('./core.js');
  const TOTAL = 1 << 16; // values carried per tensor (positions × samples)
  const TCAP = 128;       // longest sequence simulated

  // ── numerics ──
  function erf(x) { // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2)), phi = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const numel = s => s.reduce((a, b) => a * b, 1);

  // ── activations: value and derivative ──
  const sig = x => 1 / (1 + Math.exp(-x)), GK = Math.sqrt(2 / Math.PI);
  const ACT = {
    relu: [x => (x > 0 ? x : 0), x => (x > 0 ? 1 : 0)],
    sigmoid: [sig, x => { const s = sig(x); return s * (1 - s); }],
    tanh: [Math.tanh, x => { const t = Math.tanh(x); return 1 - t * t; }],
    silu: [x => x * sig(x), x => { const s = sig(x); return s * (1 + x * (1 - s)); }],
    gelu: [x => x * Phi(x), x => Phi(x) + x * phi(x)],
    gelu_tanh: [x => 0.5 * x * (1 + Math.tanh(GK * (x + 0.044715 * x * x * x))), x => { const u = GK * (x + 0.044715 * x * x * x), t = Math.tanh(u); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GK * (1 + 3 * 0.044715 * x * x); }],
    identity: [x => x, () => 1],
  };

  // ── weight variance for each init (nn.Linear weight is out × in; fan_in = in) ──
  const TRUNC2 = 1 - (2 * 2 * phi(2)) / (Phi(2) - Phi(-2)); // Var of N(0,1) truncated at ±2
  function initArg(p) { const spec = C.INITS[p.init || 'default']; const a = p.init_arg !== '' && p.init_arg != null ? +p.init_arg : spec && spec.def != null ? +spec.def : 1; return isFinite(a) ? a : 1; }
  function weightVar(p, fin, fout) {
    const k = p.init || 'default', a = initArg(p);
    switch (k) {
      case 'xavier_uniform': case 'xavier_normal': return (a * a * 2) / (fin + fout);
      case 'kaiming_uniform': case 'kaiming_normal': return 2 / fin;
      case 'orthogonal': return (a * a) / Math.max(fin, fout);
      case 'normal': return a * a;
      case 'trunc_normal': return a * a * TRUNC2;
      case 'zeros': return 0;
      default: return 1 / (3 * fin); // kaiming_uniform(a=√5): U(±1/√fan_in)
    }
  }
  function weightVarText(p, fin, fout) {
    const k = p.init || 'default', a = initArg(p), g2 = a * a === 1 ? '' : `${+(a * a).toPrecision(3)}·`;
    switch (k) {
      case 'xavier_uniform': case 'xavier_normal': return `${g2}2/(${fin}+${fout})`;
      case 'kaiming_uniform': case 'kaiming_normal': return `2/${fin}`;
      case 'orthogonal': return `${g2}1/${Math.max(fin, fout)}`;
      case 'normal': return `${a}²`;
      case 'trunc_normal': return `${+(TRUNC2).toFixed(3)}·${a}²`;
      case 'zeros': return '0';
      default: return `1/(3·${fin})`;
    }
  }
  function biasVar(p, fin) {
    if (!p.bias) return 0;
    if (p.tie) return p.zero_bias === true ? 0 : 1 / (3 * fin); // a tied layer's own bias: zeroed only when asked
    if ((p.init || 'default') !== 'default' && p.zero_bias !== false) return 0;
    return 1 / (3 * fin); // U(±1/√fan_in)
  }

  // ── names: the attribute path PyTorch would report (model.named_modules()) ──
  function linearNames(flat, order, byId) {
    const groupOf = new Map(); for (const gr of flat.groups || []) for (const id of gr.nodes) groupOf.set(id, gr);
    const scope = id => (groupOf.get(id) ? groupOf.get(id).id : '');
    const count = new Map(), seen = new Map(), names = new Map();
    for (const id of order) { const n = byId.get(id); if (n.type === 'linear' && !n.params.name) count.set(scope(id), (count.get(scope(id)) || 0) + 1); }
    for (const id of order) {
      const n = byId.get(id); if (n.type !== 'linear') continue;
      const sc = scope(id); let nm = n.params.name;
      if (!nm) { const k = (seen.get(sc) || 0) + 1; seen.set(sc, k); nm = count.get(sc) > 1 ? 'fc' + k : 'fc'; }
      const gr = groupOf.get(id);
      const pre = !gr ? '' : (String(gr.id).includes('#') ? `${gr.label}.${String(gr.id).split('#')[1]}` : gr.attr) + '.';
      names.set(id, pre + nm);
    }
    return names;
  }

  // ── the prediction ──
  // opts.sizes: overrides for size names (e.g. { t: 128 }); the batch size doesn't matter here.
  function predict(g, opts = {}) {
    // losses and their targets aren't part of the network being plotted
    if (g.nodes.some(n => n.type === 'loss')) g = C.lossParts(g).model;
    const { flat } = C.expandBlocks(g);
    const sym = Object.assign({}, flat.symbols, opts.sizes || {});
    const fl = Object.assign({}, flat, { symbols: sym });
    const { shapes, errors, conflicts } = C.inferShapes(fl);
    const ix = C.index(fl), byId = ix.byId, { order, cyclic } = C.topo(fl, ix);
    if (cyclic.length) return { error: 'The graph has a cycle.' };
    if (conflicts.size) return { error: 'Fix the size conflicts first (marked in red).' };
    if (errors.size) { const [id, m] = [...errors][0]; return { error: m, node: id }; }
    const shapeOf = id => { const s = shapes.get(id); if (!s) return null; const v = s.map(d => C.evalDim(d, sym)); return v.every(x => Number.isFinite(x)) ? v : null; };
    const consumers = id => (ix.outs.get(id) || []).map(o => byId.get(o.to)).filter(Boolean);
    const notes = new Set();

    // ── positions: sizes that index the sequence (input dims other than the batch and the features) ──
    const posSyms = new Set();
    for (const n of fl.nodes) if (n.type === 'input') {
      const d = C.parseShape(n.params.shape), emb = consumers(n.id).some(c => c.type === 'embed');
      (emb ? d.slice(1) : d.slice(1, -1)).forEach(x => posSyms.add(String(x)));
    }
    let Tdata = 1; for (const s of posSyms) { const v = C.evalDim(s, sym); if (v > Tdata) Tdata = v; }
    const TS = Math.max(1, Math.min(Tdata, TCAP));
    if (Tdata > TCAP) notes.add(`Sequences are simulated with their first ${TCAP} positions.`);
    const posT = id => { const s = shapes.get(id); return s && s.length >= 2 && posSyms.has(String(s[s.length - 2])) ? TS : 1; };
    let Dh = 0; for (const n of fl.nodes) if (n.type === 'heads') { const s = shapeOf(n.id); if (s) Dh = Math.max(Dh, s[s.length - 1]); }
    let NS = Math.max(256, Math.floor(TOTAL / TS)); if (Dh > 0) NS = Math.max(Dh, Math.floor(NS / Dh) * Dh);

    // ── random numbers and tensors ──
    const rand = mulberry(0x5eed); let spare = null;
    const gauss = () => { if (spare !== null) { const v = spare; spare = null; return v; } let u = 0; while (u === 0) u = rand(); const r = Math.sqrt(-2 * Math.log(u)), a = 2 * Math.PI * rand(); spare = r * Math.sin(a); return r * Math.cos(a); };
    const feat = (T, X) => ({ k: 'f', T, X });
    // independent N(0, v) values, rescaled to exactly mean 0 and variance v (less sampling noise)
    const iid = (T, v) => {
      const X = new Float32Array(T * NS); let m = 0, q = 0;
      for (let i = 0; i < X.length; i++) { const z = gauss(); X[i] = z; m += z; }
      m /= X.length; for (let i = 0; i < X.length; i++) { X[i] -= m; q += X[i] * X[i]; }
      const k = q > 0 ? Math.sqrt(Math.max(0, v) / (q / X.length)) : 0; for (let i = 0; i < X.length; i++) X[i] *= k;
      return feat(T, X);
    };
    // E[x_t x_u] over the samples (T × T)
    const moments = A => {
      const T = A.T, X = A.X, M = new Float64Array(T * T);
      for (let t = 0; t < T; t++) for (let u = 0; u <= t; u++) { let s = 0; const a = t * NS, b = u * NS; for (let i = 0; i < NS; i++) s += X[a + i] * X[b + i]; M[t * T + u] = M[u * T + t] = s / NS; }
      return M;
    };
    // fresh Gaussian with covariance K between positions (Cholesky, with a little jitter if needed)
    const withCov = (T, K) => {
      if (T === 1) return iid(1, K[0]);
      let L, jit = 1e-10 * Math.max(1e-30, ...Array.from({ length: T }, (_, t) => K[t * T + t]));
      for (let tries = 0; tries < 8; tries++) {
        L = new Float64Array(T * T); let ok = true;
        for (let i = 0; i < T && ok; i++) for (let j = 0; j <= i; j++) {
          let s = K[i * T + j] + (i === j ? jit : 0); for (let k = 0; k < j; k++) s -= L[i * T + k] * L[j * T + k];
          if (i === j) { if (s <= 0) { ok = false; break; } L[i * T + i] = Math.sqrt(s); } else L[i * T + j] = s / L[j * T + j];
        }
        if (ok) break; jit *= 100; if (tries === 7) return iid(T, 0);
      }
      const X = new Float32Array(T * NS), z = new Float64Array(T);
      for (let s = 0; s < NS; s++) { for (let t = 0; t < T; t++) z[t] = gauss(); for (let t = 0; t < T; t++) { let v = 0; for (let u = 0; u <= t; u++) v += L[t * T + u] * z[u]; X[t * NS + s] = v; } }
      // rescale each position to its exact target variance (correlated positions don't average sampling noise away)
      for (let t = 0; t < T; t++) { let q = 0; for (let s = 0; s < NS; s++) q += X[t * NS + s] * X[t * NS + s]; const k = q > 0 ? Math.sqrt(K[t * T + t] / (q / NS)) : 0; for (let s = 0; s < NS; s++) X[t * NS + s] *= k; }
      return feat(T, X);
    };
    const meanAll = X => { let s = 0; for (let i = 0; i < X.length; i++) s += X[i]; return s / X.length; };
    const varAll = X => { const m = meanAll(X); let s = 0; for (let i = 0; i < X.length; i++) { const d = X[i] - m; s += d * d; } return s / X.length; };
    const mapX = (A, f) => { const X = new Float32Array(A.X.length); for (let i = 0; i < X.length; i++) X[i] = f(A.X[i], i); return Object.assign({}, A, { X }); };
    const like = (A, X) => Object.assign({}, A, { X });
    const names = linearNames(fl, order, byId);
    // a tied layer uses the weights (so the init) of the layer it shares with
    const ownVar = o => {
      const q = o.params;
      if (o.type === 'embed') { const nm = C.evalDim(q.num, sym), dm = C.evalDim(q.dim, sym); return (q.init || 'default') === 'default' ? { v: 1, text: '1' } : { v: weightVar(q, dm, nm), text: weightVarText(q, dm, nm) }; }
      const fi = C.evalDim(q.in, sym), fo = C.evalDim(q.out, sym); return { v: weightVar(q, fi, fo), text: weightVarText(q, fi, fo) };
    };
    const tiedVar = n => { const o = C.tieOwner(fl, n, byId); if (!o) return null; const r = ownVar(o); return { v: r.v, text: `${r.text}, tied to ${o.params.title || names.get(o.id) || o.params.name || o.type}` }; };
    const val = new Map(), tape = new Map(), layers = [];
    const unsupported = (n, what) => ({ error: `Can't predict through ${what || n.params.fn || n.type}: it isn't one of the operations the variance plot knows.`, node: n.id });

    for (const id of order) {
      const n = byId.get(id), p = n.params, shape = shapeOf(id);
      const ins = (ix.ins.get(id) || []).map(s => val.get(s));
      const A = ins[0];
      if (n.type !== 'input' && n.type !== 'embed' && ins.some(v => !v || v.k === 'idx')) {
        if (ins.some(v => v && v.k === 'idx')) return { error: 'Token indices can only go into an Embedding.', node: id };
        return { error: 'A layer has an unconnected input.', node: id };
      }
      let out = null;
      switch (n.type) {
        case 'input': {
          const embs = consumers(id).filter(c => c.type === 'embed');
          if (!embs.length) { out = iid(posT(id), 1); break; }
          // token ids for one sequence, uniform over the smallest table they index
          const num = Math.min(...embs.filter(c => !c.params.positions).map(c => C.evalDim(c.params.num, sym)).filter(v => v > 0), 1e9);
          const sh = shapes.get(id), T = sh && posSyms.has(String(sh[sh.length - 1])) ? TS : 1; // token ids are (batch × positions)
          const ids = Int32Array.from({ length: T }, () => Math.floor(rand() * (num < 1e9 ? num : 1e9)));
          out = { k: 'idx', ids };
          break;
        }
        case 'output': out = A; break;
        case 'linear': {
          if (A.k !== 'f') return unsupported(n, 'a Linear on attention scores');
          const fin = A.shape ? A.shape[A.shape.length - 1] : C.evalDim(p.in, sym), fout = C.evalDim(p.out, sym);
          const own = tiedVar(n), vw = own ? own.v : weightVar(p, fin, fout), vb = biasVar(p, fin), M = moments(A);
          for (let i = 0; i < M.length; i++) M[i] = fin * vw * M[i] + vb;
          out = withCov(A.T, M);
          tape.set(id, { fout, vw });
          layers.push({ id, name: names.get(id), title: p.title || '', init: p.init || 'default', wvar: vw, wvarText: own ? own.text : weightVarText(p, fin, fout), bias: !!p.bias, fin, fout, fwdIn: varAll(A.X), fwdOut: varAll(out.X), bwdIn: null, bwdOut: null });
          break;
        }
        case 'norm': {
          const def = p.kind === 'RMSNorm' ? 1.1920928955078125e-7 : 1e-5, eps = p.eps !== '' && p.eps != null && isFinite(+p.eps) ? +p.eps : def;
          const perPos = p.kind === 'LayerNorm' || p.kind === 'RMSNorm', rms = p.kind === 'RMSNorm';
          const T = A.T, X = new Float32Array(A.X.length), scale = new Float64Array(T);
          const sets = perPos ? Array.from({ length: T }, (_, t) => [t * NS, (t + 1) * NS]) : [[0, T * NS]];
          for (const [a, b] of sets) {
            let m = 0, q = 0; for (let i = a; i < b; i++) { m += A.X[i]; q += A.X[i] * A.X[i]; }
            m /= b - a; q /= b - a;
            const s = rms ? 1 / Math.sqrt(q + eps) : 1 / Math.sqrt(Math.max(0, q - m * m) + eps), mu = rms ? 0 : m;
            for (let i = a; i < b; i++) X[i] = (A.X[i] - mu) * s;
            for (let t = Math.floor(a / NS); t < Math.ceil(b / NS); t++) scale[t] = s;
          }
          out = feat(T, X); tape.set(id, { sets, scale, rms });
          break;
        }
        case 'act': {
          if (p.fn === 'softmax') { out = softmaxFwd(id, A, shape, p.dim); if (out.error) return out; break; }
          const f = ACT[p.fn]; if (!f) return unsupported(n, p.fn);
          out = mapX(A, f[0]); tape.set(id, { d: f[1] });
          break;
        }
        case 'dropout': out = A; notes.add('Dropout is treated as off (eval mode).'); break;
        case 'embed': {
          const nm = C.evalDim(p.num, sym), dm = C.evalDim(p.dim, sym);
          const own = tiedVar(n), v = own ? own.v : (p.init || 'default') === 'default' ? 1 : weightVar(p, dm, nm), T = posT(id);
          if (p.positions || T === 1 || !A || !A.ids) { out = iid(T, v); break; } // each position has its own row
          // positions holding the same token read the same row (this matters for small tables, such as BERT's two token types)
          const rows = new Map(), X = new Float32Array(T * NS);
          for (let t = 0; t < T; t++) { const k = A.ids[t % A.ids.length]; if (!rows.has(k)) rows.set(k, iid(1, v).X); X.set(rows.get(k), t * NS); }
          out = feat(T, X);
          break;
        }
        case 'heads': case 'merge': out = A; break;
        case 'transpose':
          if (A.k === 'f') out = Object.assign({}, A, { k: 'fT' });
          else if (A.k === 'fT') out = Object.assign({}, A, { k: 'f' });
          else out = transposeScores(A);
          break;
        case 'scale': { const Dl = A.shape ? A.shape[A.shape.length - 1] : 1, k = 1 / Math.sqrt(Dl); out = mapX(A, x => x * k); tape.set(id, { k }); break; }
        case 'rope': {
          if (p.mode === 'axial' || p.mode === 'mrope') return unsupported(n, 'rotary embeddings at given positions');
          if (A.k !== 'f') return unsupported(n, 'rotary embeddings outside the features');
          const r = ropeTables(A, +p.base || 10000), X = new Float32Array(A.X.length);
          for (let t = 0; t < A.T; t++) for (let s = 0; s < NS; s++) {
            const q = r.pair(s), o = t * NS; if (q < 0) { X[o + s] = A.X[o + s]; continue; }
            const c = r.cos(t, s), sn = r.sin(t, s);
            X[o + s] = r.first(s) ? A.X[o + s] * c - A.X[o + q] * sn : A.X[o + s] * c + A.X[o + q] * sn;
          }
          out = feat(A.T, X); tape.set(id, { rope: r });
          break;
        }
        case 'mask': {
          if (p.mode === 'given') return unsupported(n, 'an attention mask given as an input');
          if (A.k !== 's') return unsupported(n, 'a causal mask outside attention scores');
          const T = A.T; out = mapX(A, (x, i) => (i % T > Math.floor(i / T) % T ? -Infinity : x));
          break;
        }
        case 'binop': {
          const op = p.op || '+';
          if (op === '@') { out = matmulFwd(id, ins[0], ins[1]); if (out.error) return out; break; }
          const f = { '+': (x, y) => x + y, '-': (x, y) => x - y, '*': (x, y) => x * y, '/': (x, y) => x / y }[op];
          if (!f) return unsupported(n, `the operator ${op}`);
          const B = ins[1];
          if (!B) {
            let c = Number(p.const); if (!Number.isFinite(c)) c = C.evalDim(p.const, sym);
            if (!Number.isFinite(c)) return unsupported(n, `the constant "${p.const}"`);
            out = mapX(A, p.rev ? x => f(c, x) : x => f(x, c)); tape.set(id, { c });
            break;
          }
          if (A.k !== B.k || (A.k === 's' && (A.T !== B.T || A.Hs !== B.Hs))) return unsupported(n, `${op} between attention scores and features`);
          if (A.k === 's') { out = like(A, A.X.map((x, i) => f(x, B.X[i]))); break; }
          const T = Math.max(A.T, B.T), X = new Float32Array(T * NS);
          for (let t = 0; t < T; t++) { const a = (A.T === 1 ? 0 : t) * NS, b = (B.T === 1 ? 0 : t) * NS, o = t * NS; for (let s = 0; s < NS; s++) X[o + s] = f(A.X[a + s], B.X[b + s]); }
          out = feat(T, X);
          break;
        }
        case 'func': {
          const fn = String(p.init || p.fn || ''), base = fn.replace(/\(.*$/, '').replace(/^.*\./, '').toLowerCase();
          if (/^(matmul|mm|bmm)$/.test(base) && ins.length === 2) { out = matmulFwd(id, ins[0], ins[1]); if (out.error) return out; break; }
          if (/^(cat|concat|concatenate|stack)$/.test(base) && ins.length > 1) {
            if (ins.some(v => v.k !== 'f')) return unsupported(n, fn + ' of attention scores');
            // each feature coordinate comes from one part, with probability equal to that part's share
            const w = ins.map(v => numel(v.shape || [1]) / Math.max(1, v.T)), tot = w.reduce((x, y) => x + y, 0);
            const T = Math.max(...ins.map(v => v.T)), X = new Float32Array(T * NS), pick = new Int32Array(NS);
            for (let s = 0; s < NS; s++) { let u = rand() * tot, k = 0; while (k < w.length - 1 && u >= w[k]) { u -= w[k]; k++; } pick[s] = k; }
            for (let t = 0; t < T; t++) for (let s = 0; s < NS; s++) { const v = ins[pick[s]]; X[t * NS + s] = v.X[(v.T === 1 ? 0 : t) * NS + s]; }
            out = feat(T, X); tape.set(id, { cat: true });
            break;
          }
          if (base === 'softmax') { const m = String(p.extra || '').match(/dim\s*=\s*(-?\d+)/); out = softmaxFwd(id, A, shape, m ? m[1] : -1); if (out.error) return out; break; }
          const act = { relu: 'relu', gelu: 'gelu', tanh: 'tanh', sigmoid: 'sigmoid', silu: 'silu', identity: 'identity', dropout: 'identity', contiguous: 'identity', reshape: 'identity', view: 'identity', flatten: 'identity', squeeze: 'identity', unsqueeze: 'identity' }[base];
          if (!act || ins.length !== 1) return unsupported(n, fn);
          out = mapX(A, ACT[act][0]); tape.set(id, { d: ACT[act][1] });
          break;
        }
        default: return unsupported(n);
      }
      if (out.k !== 'idx') out = Object.assign({}, out, { shape });
      val.set(id, out);
    }
    if (TS > 1) notes.add(`Attention is computed on ${TS} positions with ${Math.max(1, Math.floor(NS / Math.max(1, Dh || NS)))} simulated heads.`);

    // ── backward: a standard Gaussian gradient enters at every output ──
    const grad = new Map();
    const addGrad = (id, G) => {
      if (!id || !G) return; const v = val.get(id); if (!v || v.k === 'idx') return;
      // a tensor broadcast over positions receives the sum of their gradients
      if (v.k !== 's' && v.T === 1 && G.length > NS) { const S = new Float32Array(NS); for (let i = 0; i < G.length; i++) S[i % NS] += G[i]; G = S; }
      const cur = grad.get(id);
      if (!cur) grad.set(id, G); else for (let i = 0; i < cur.length; i++) cur[i] += G[i];
    };
    const mapG = (G, f) => { const Y = new Float32Array(G.length); for (let i = 0; i < G.length; i++) Y[i] = f(G[i], i); return Y; };
    for (const n of fl.nodes) if (n.type === 'output' && val.get(n.id) && val.get(n.id).k !== 'idx') { const v = val.get(n.id); grad.set(n.id, iid(v.T, 1).X); }
    for (let k = order.length - 1; k >= 0; k--) {
      const id = order[k], n = byId.get(id), p = n.params, G = grad.get(id); if (!G) continue;
      grad.delete(id);
      const src = ix.ins.get(id) || [], ins = src.map(s => val.get(s)), A = ins[0], me = val.get(id), t = tape.get(id);
      switch (n.type) {
        case 'output': case 'dropout': case 'heads': case 'merge': addGrad(src[0], G); break;
        case 'transpose': addGrad(src[0], me.k === 's' ? transposeScores(like(me, G)).X : G); break;
        case 'mask': addGrad(src[0], mapG(G, (g, i) => (me.X[i] === -Infinity ? 0 : g))); break;
        case 'linear': {
          const M = moments(feat(me.T, G));
          for (let i = 0; i < M.length; i++) M[i] *= t.fout * t.vw;
          const dx = withCov(me.T, M).X;
          const L = layers.find(l => l.id === id); L.bwdOut = varAll(G); L.bwdIn = varAll(dx);
          addGrad(src[0], dx);
          break;
        }
        case 'norm': {
          const Y = me.X, dX = new Float32Array(G.length);
          for (const [a, b] of t.sets) {
            const m = b - a; let mg = 0, mgy = 0; for (let i = a; i < b; i++) { mg += G[i]; mgy += G[i] * Y[i]; } mg /= m; mgy /= m;
            const s = t.scale[Math.floor(a / NS)];
            for (let i = a; i < b; i++) dX[i] = s * (G[i] - (t.rms ? 0 : mg) - Y[i] * mgy);
          }
          addGrad(src[0], dX);
          break;
        }
        case 'scale': addGrad(src[0], mapG(G, g => g * t.k)); break;
        case 'rope': { // the inverse rotation
          const r = t.rope, dX = new Float32Array(G.length);
          for (let tt = 0; tt < me.T; tt++) for (let s = 0; s < NS; s++) {
            const q = r.pair(s), o = tt * NS; if (q < 0) { dX[o + s] = G[o + s]; continue; }
            const c = r.cos(tt, s), sn = r.sin(tt, s);
            dX[o + s] = r.first(s) ? G[o + s] * c + G[o + q] * sn : G[o + s] * c - G[o + q] * sn;
          }
          addGrad(src[0], dX); break;
        }
        case 'act': case 'func': {
          if (t && t.sm) { addGrad(src[0], softmaxBwd(t.sm, me, G)); break; }
          if (t && t.mm) { matmulBwd(t.mm, src, G); break; }
          if (t && t.cat) {
            for (const s of src) { // each part's gradient comes from different downstream weights: same distribution, independent
              const perm = Int32Array.from({ length: NS }, (_, i) => i); for (let i = NS - 1; i > 0; i--) { const j = (rand() * (i + 1)) | 0; const q = perm[i]; perm[i] = perm[j]; perm[j] = q; }
              const T = me.T, Y = new Float32Array(G.length); for (let tt = 0; tt < T; tt++) for (let i = 0; i < NS; i++) Y[tt * NS + i] = G[tt * NS + perm[i]];
              addGrad(s, Y);
            }
            break;
          }
          if (t && t.d) { const X = A.X; addGrad(src[0], mapG(G, (g, i) => g * t.d(X[i]))); }
          break;
        }
        case 'binop': {
          const op = p.op || '+';
          if (op === '@') { matmulBwd(t.mm, src, G); break; }
          const B = ins[1];
          if (!B) {
            const c = t.c, X = A.X;
            const d = { '+': () => 1, '-': () => (p.rev ? -1 : 1), '*': () => c, '/': i => (p.rev ? -c / (X[i] * X[i]) : 1 / c) }[op];
            addGrad(src[0], mapG(G, (g, i) => g * d(i)));
            break;
          }
          const T = me.k === 's' ? 1 : me.T, at = i => (A.k === 's' || A.T === T ? i : i % NS), bt = i => (B.k === 's' || B.T === T ? i : i % NS);
          if (op === '+') { addGrad(src[0], G); addGrad(src[1], G); }
          else if (op === '-') { addGrad(src[0], G); addGrad(src[1], mapG(G, g => -g)); }
          else if (op === '*') { addGrad(src[0], mapG(G, (g, i) => g * B.X[bt(i)])); addGrad(src[1], mapG(G, (g, i) => g * A.X[at(i)])); }
          else if (op === '/') { addGrad(src[0], mapG(G, (g, i) => g / B.X[bt(i)])); addGrad(src[1], mapG(G, (g, i) => (-g * A.X[at(i)]) / (B.X[bt(i)] * B.X[bt(i)]))); }
          break;
        }
        default: break;
      }
    }
    const inputs = fl.nodes.filter(n => n.type === 'input').map(n => ({ id: n.id, name: n.params.name, idx: (val.get(n.id) || {}).k === 'idx' }));
    return { layers, notes: [...notes], inputs, samples: NS, positions: TS };

    // ── rotary embeddings: within each simulated head (a group of head-size sample columns, the same
    // grouping attention uses), feature i turns together with feature i + D/2 by angle t·base^(−2i/D)
    function ropeTables(A, base) {
      const D = A.shape[A.shape.length - 1], Dsim = Math.min(D, NS), half = Math.floor(Dsim / 2);
      const inv = new Float64Array(half); for (let i = 0; i < half; i++) inv[i] = Math.pow(base, (-2 * (i * (D / 2) / half)) / D);
      const j = s => s % Dsim;
      return {
        pair: s => { const k = j(s); if (k >= 2 * half) return -1; return k < half ? s + half : s - half; },
        first: s => j(s) < half,
        cos: (t, s) => Math.cos(t * inv[j(s) % half]),
        sin: (t, s) => Math.sin(t * inv[j(s) % half]),
      };
    }
    // ── attention scores: per simulated head h, a (T × T) matrix, stored [h][t][u] ──
    function transposeScores(A) {
      const T = A.T, X = new Float32Array(A.X.length);
      for (let h = 0; h < A.Hs; h++) for (let t = 0; t < T; t++) for (let u = 0; u < T; u++) X[h * T * T + u * T + t] = A.X[h * T * T + t * T + u];
      return like(A, X);
    }
    // q @ kᵀ → scores; scores @ v → features; anything else between plain features (no positions) → fresh Gaussian
    function matmulFwd(id, A, B) {
      if (A.k === 'f' && B.k === 'fT') {
        if (A.T !== B.T) return { error: "Can't predict attention between sequences of different lengths." };
        const Dk = A.shape[A.shape.length - 1], Dsim = Math.min(Dk, NS), Hs = Math.max(1, Math.floor(NS / Dsim)), fac = Math.sqrt(Dk / Dsim), T = A.T;
        const X = new Float32Array(Hs * T * T);
        for (let h = 0; h < Hs; h++) for (let t = 0; t < T; t++) for (let u = 0; u < T; u++) {
          let s = 0; const a = t * NS + h * Dsim, b = u * NS + h * Dsim; for (let d = 0; d < Dsim; d++) s += A.X[a + d] * B.X[b + d];
          X[(h * T + t) * T + u] = s * fac;
        }
        tape.set(id, { mm: { kind: 'qk', A, B, Dsim, Hs, fac, T } });
        return { k: 's', T, Hs, Dsim, X };
      }
      if (A.k === 's' && B.k === 'f') {
        const T = A.T, Hs = A.Hs, Dsim = Math.max(1, Math.floor(NS / Hs)), X = new Float32Array(T * NS);
        const hOf = s => Math.min(Hs - 1, Math.floor(s / Dsim));
        for (let t = 0; t < T; t++) for (let s = 0; s < NS; s++) {
          const h = hOf(s), P = (h * T + t) * T; let v = 0;
          for (let u = 0; u < T; u++) { const pw = A.X[P + u]; if (pw) v += pw * B.X[(B.T === 1 ? 0 : u) * NS + s]; }
          X[t * NS + s] = v;
        }
        tape.set(id, { mm: { kind: 'pv', A, B, Hs, Dsim, T } });
        return feat(T, X);
      }
      if (A.k === 'f' && B.k === 'f' && A.T === 1 && B.T === 1) {
        const K = A.shape[A.shape.length - 1]; let qa = 0, qb = 0; for (let i = 0; i < NS; i++) { qa += A.X[i] * A.X[i]; qb += B.X[i] * B.X[i]; }
        qa /= NS; qb /= NS;
        const M = B.shape.length > 1 ? B.shape[B.shape.length - 1] : 1, R = A.shape.length > 1 ? A.shape[A.shape.length - 2] : 1;
        tape.set(id, { mm: { kind: 'plain', qa, qb, N: M, R } });
        return iid(1, K * qa * qb);
      }
      return { error: "This matrix multiply isn't one the variance plot can follow (it supports attention: q @ kᵀ and weights @ v)." };
    }
    function matmulBwd(mm, src, G) {
      if (mm.kind === 'qk') {
        const { A, B, Dsim, Hs, fac, T } = mm, dA = new Float32Array(A.X.length), dB = new Float32Array(B.X.length);
        for (let h = 0; h < Hs; h++) for (let t = 0; t < T; t++) for (let u = 0; u < T; u++) {
          const g = G[(h * T + t) * T + u] * fac; if (!g) continue;
          const a = t * NS + h * Dsim, b = u * NS + h * Dsim;
          for (let d = 0; d < Dsim; d++) { dA[a + d] += g * B.X[b + d]; dB[b + d] += g * A.X[a + d]; }
        }
        addGrad(src[0], dA); addGrad(src[1], dB);
      } else if (mm.kind === 'pv') {
        const { A, B, Hs, Dsim, T } = mm, dP = new Float32Array(A.X.length), dV = new Float32Array(B.X.length);
        for (let t = 0; t < T; t++) for (let s = 0; s < NS; s++) {
          const g = G[t * NS + s]; if (!g) continue;
          const h = Math.min(Hs - 1, Math.floor(s / Dsim)), P = (h * T + t) * T;
          for (let u = 0; u < T; u++) { const bi = (B.T === 1 ? 0 : u) * NS + s; dP[P + u] += g * B.X[bi]; dV[bi] += g * A.X[P + u]; }
        }
        addGrad(src[0], dP); addGrad(src[1], dV);
      } else {
        let qg = 0; for (let i = 0; i < G.length; i++) qg += G[i] * G[i]; qg /= G.length;
        addGrad(src[0], iid(1, mm.N * qg * mm.qb).X); addGrad(src[1], iid(1, mm.R * qg * mm.qa).X);
      }
    }
    // softmax over attention scores (rows), or over features (rows made of feature coordinates at one position)
    function softmaxFwd(id, A, shape, dim) {
      const r = shape.length, ax = ((+dim % r) + r) % r;
      if (ax !== r - 1) return { error: 'The variance plot supports softmax over the last dimension only.' };
      if (A.k === 's') {
        const T = A.T, X = new Float32Array(A.X.length);
        for (let row = 0; row < A.Hs * T; row++) {
          const o = row * T; let mx = -Infinity; for (let u = 0; u < T; u++) mx = Math.max(mx, A.X[o + u]);
          let z = 0; for (let u = 0; u < T; u++) { const e = A.X[o + u] === -Infinity ? 0 : Math.exp(A.X[o + u] - mx); X[o + u] = e; z += e; }
          for (let u = 0; u < T; u++) X[o + u] /= z;
        }
        tape.set(id, { sm: { kind: 's' } });
        return like(A, X);
      }
      const n = shape[ax], m = Math.min(n, NS), f = m / n, T = A.T, X = new Float32Array(A.X.length);
      const rows = [];
      for (let t = 0; t < T; t++) for (let k = 0; k * m < NS; k++) {
        const idx = []; for (let j = 0; j < m; j++) idx.push(t * NS + ((k * m + j) % NS));
        let mx = -Infinity; for (const i of idx) mx = Math.max(mx, A.X[i]);
        const e = idx.map(i => Math.exp(A.X[i] - mx)), z = e.reduce((x, y) => x + y, 0);
        idx.forEach((i, j) => { X[i] = (e[j] / z) * f; });
        rows.push(idx);
      }
      tape.set(id, { sm: { kind: 'f', rows } });
      return like(A, X);
    }
    function softmaxBwd(sm, me, G) {
      const P = me.X, dX = new Float32Array(G.length);
      if (sm.kind === 's') {
        const T = me.T;
        for (let row = 0; row < me.Hs * T; row++) { const o = row * T; let dot = 0; for (let u = 0; u < T; u++) dot += P[o + u] * G[o + u]; for (let u = 0; u < T; u++) dX[o + u] = P[o + u] * (G[o + u] - dot); }
      } else for (const idx of sm.rows) { let dot = 0; for (const i of idx) dot += P[i] * G[i]; for (const i of idx) dX[i] = P[i] * (G[i] - dot); }
      return dX;
    }
  }

  return { predict, weightVar, biasVar };
})();
if (typeof module !== 'undefined') module.exports = NNVar;
