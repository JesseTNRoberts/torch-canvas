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
// Images (e × C × H × W) are carried the same way with pixels as the positions. A convolution is a Linear over each
// k × k window: its output is a fresh Gaussian whose covariance between pixels P and Q is
//     C_in · Var(w) · Σ_taps E[x_(sP+δ) x_(sQ+δ)] + Var(b)          (taps that fall in the padding count 0)
// Feature maps up to 8 × 8 are simulated pixel by pixel (borders and padding exact); larger ones as an 8 × 8
// patch that wraps around (stationary statistics), with the padding's effect at the borders averaged in.
// Runs in the browser and in Node.
const NNVar = (() => {
  'use strict';
  const C = typeof NN !== 'undefined' ? NN : require('./core.js');
  const TOTAL = 1 << 16; // values carried per tensor (positions × samples)
  const TCAP = 128;       // longest sequence simulated
  const KMAX = 8;         // entries carried along an inner axis (Mamba's state: d_state of them per unit)
  const SEQCAP = 256;     // longest sequence simulated from a flattened image (ViT: 196 patches)

  // ── numerics ──
  function erf(x) { // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2)), phi = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  // the standard normal quantile (Acklam; relative error < 1.2e-9)
  function probit(p) {
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    const q = p - 0.5, r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
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
    softplus: [x => (x > 20 ? x : Math.log1p(Math.exp(x))), sig],
    exp: [Math.exp, Math.exp],
  };
  // loops through a Delay (strongly connected parts of the graph that contain one); a loop of elementwise ops only
  // (Mamba's h = Ā ⊙ h_prev + B̄u) is a scan: it is run along the simulated positions, step by step, forward and back
  const SCAN_FUNCS = /^(exp|identity|contiguous|unsqueeze|squeeze|relu|gelu|tanh|sigmoid|silu)$/;
  function loopsOf(fl, ix) {
    const idx = new Map(), low = new Map(), on = new Set(), st = [], comps = []; let k = 0;
    const visit = v => {
      idx.set(v, k); low.set(v, k); k++; st.push(v); on.add(v);
      for (const o of ix.outs.get(v) || []) { const w = o.to; if (!idx.has(w)) { visit(w); low.set(v, Math.min(low.get(v), low.get(w))); } else if (on.has(w)) low.set(v, Math.min(low.get(v), idx.get(w))); }
      if (low.get(v) === idx.get(v)) { const c = []; let w; do { w = st.pop(); on.delete(w); c.push(w); } while (w !== v); if (c.length > 1) comps.push(c); }
    };
    for (const n of fl.nodes) if (!idx.has(n.id)) visit(n.id);
    return comps.filter(c => c.some(id => ix.byId.get(id).type === 'delay')).map(c => ({
      members: new Set(c),
      scan: c.every(id => { const n = ix.byId.get(id), p = n.params; if (n.type === 'delay') return true; if (n.type === 'binop') return (p.op || '+') !== '@'; if (n.type === 'act') return p.fn !== 'softmax'; if (n.type === 'func') return !p.init && SCAN_FUNCS.test(String(p.fn).replace(/^.*\./, '')); return false; }),
    }));
  }

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
    if (!p.bias || p.bias_init === 'dt') return 0;
    if (p.tie) return p.zero_bias === true ? 0 : 1 / (3 * fin); // a tied layer's own bias: zeroed only when asked
    if ((p.init || 'default') !== 'default' && p.zero_bias !== false) return 0;
    return 1 / (3 * fin); // U(±1/√fan_in)
  }

  // ── names: the attribute path PyTorch would report (model.named_modules()) ──
  function linearNames(flat, order, byId) {
    const groupOf = new Map(); for (const gr of flat.groups || []) for (const id of gr.nodes) groupOf.set(id, gr);
    // a block used inside a block: its group's id is the use node, listed among the outer group's layers
    const parentOf = new Map(), byUse = new Map();
    for (const gr of flat.groups || []) { const u = String(gr.id).replace(/#\d+$/, ''); if (!byUse.has(u)) byUse.set(u, []); byUse.get(u).push(gr); }
    for (const gr of flat.groups || []) for (const id of gr.nodes) for (const ch of byUse.get(id) || []) if (ch !== gr) parentOf.set(ch, gr);
    const scope = id => (groupOf.get(id) ? groupOf.get(id).id : '');
    const count = new Map(), seen = new Map(), names = new Map(), W = { linear: 'fc', conv: 'conv', conv1d: 'conv1d' };
    for (const id of order) { const n = byId.get(id); if (W[n.type] && !n.params.name) { const k = scope(id) + '|' + n.type; count.set(k, (count.get(k) || 0) + 1); } }
    for (const id of order) {
      const n = byId.get(id); if (!W[n.type]) continue;
      const sc = scope(id) + '|' + n.type; let nm = n.params.name;
      if (!nm) { const k = (seen.get(sc) || 0) + 1; seen.set(sc, k); nm = count.get(sc) > 1 ? W[n.type] + k : W[n.type]; }
      // the path through every block use around it, outermost first (stage2.0.conv1.conv)
      let pre = '';
      for (let gr = groupOf.get(id), k = 0; gr && k < 20; gr = parentOf.get(gr), k++) pre = (String(gr.id).includes('#') ? `${gr.label}.${String(gr.id).split('#')[1]}` : gr.attr) + '.' + pre;
      names.set(id, pre + nm);
    }
    return names;
  }

  // ── the prediction ──
  // opts.sizes: overrides for size names (e.g. { t: 128 }); the batch size doesn't matter here.
  // A loop through a Delay (a recurrence) is looked at one step at a time: each Delay in a loop hands the step a
  // previous state that is independent of the current input (as in a real network, where it came from earlier
  // inputs), and gradients stop there (no backpropagation through time). The state's distribution is the one the
  // step settles to: starting from zeros, the step is repeated, each time taking the states it just produced, until
  // their variance stops changing.
  function predict(g, opts = {}) {
    if (opts._state !== undefined || !g.nodes.some(n => n.type === 'delay')) return predictStep(g, opts);
    { const fl = C.expandBlocks(g.nodes.some(n => n.type === 'loss') ? C.lossParts(g).model : g).flat; if (loopsOf(fl, C.index(fl)).every(L => L.scan)) return predictStep(g, opts); }
    let state = null, r = null, prev = null, k = 0;
    const persist = new Map(); // each layer's fixed part, the same at every step (weights don't change over time)
    for (; k < 60; k++) {
      r = predictStep(g, Object.assign({}, opts, { _state: state, _fwdOnly: true, _persist: persist, _seed: k + 1 }));
      if (r.error || !r.nextState) return r.error ? r : predictStep(g, Object.assign({}, opts, { _state: null }));
      const v = [...r.nextState.values()].map(X => { let q = 0; for (let i = 0; i < X.length; i++) q += X[i] * X[i]; return q / X.length; });
      state = r.nextState;
      if (prev && v.every((x, i) => Math.abs(x - prev[i]) <= 0.002 * Math.max(x, prev[i], 1e-12))) break;
      prev = v;
    }
    const out = predictStep(g, Object.assign({}, opts, { _state: state, _persist: persist, _seed: k + 2 }));
    if (!out.error) out.notes = [`A recurrence is shown for one step: the previous state comes from a Delay, independent of the current input, with the distribution it settles to (found by repeating the step ${k + 1} times from zeros); gradients stop at the Delay (no backpropagation through time).`, ...out.notes];
    return out;
  }
  function predictStep(g, opts = {}) {
    // losses and their targets aren't part of the network being plotted
    if (g.nodes.some(n => n.type === 'loss')) g = C.lossParts(g).model;
    const { flat, origin } = C.expandBlocks(g);
    // followers (EMA ties) run without gradients: nothing flows back into them or through them
    const followUses = new Set(g.nodes.filter(n => n.type === 'block' && n.params.follows).map(n => n.id));
    const noGrad = new Set(flat.nodes.filter(n => n.params.follows || (origin.get(n.id) && followUses.has(origin.get(n.id).use))).map(n => n.id));
    const sym = Object.assign({}, flat.symbols, opts.sizes || {});
    const fl = Object.assign({}, flat, { symbols: sym });
    const { shapes, errors, conflicts } = C.inferShapes(fl);
    const ix = C.index(fl), byId = ix.byId, { order: order0, cyclic } = C.topo(fl, ix);
    if (cyclic.length) return { error: 'The graph has a cycle.' };
    // a Delay outside any loop is a shift along time and comes after its input; a loop through a Delay (a recurrence)
    // isn't simulated yet
    let order = order0, scans = [];
    const loops = fl.nodes.some(n => n.type === 'delay') ? loopsOf(fl, ix) : [];
    if (loops.some(L => L.scan)) {
      // scan loops: their members together, after everything they read and before everything that reads them
      scans = loops.filter(L => L.scan); const compOf = new Map(); scans.forEach((L, i) => L.members.forEach(id => compOf.set(id, 'L' + i)));
      const other = new Set(loops.filter(L => !L.scan).flatMap(L => [...L.members].filter(id => byId.get(id).type === 'delay')));
      const key = id => compOf.get(id) || id, pos = new Map(order0.map((id, i) => [id, i]));
      const keys = [...new Set(order0.map(key))], indeg = new Map(keys.map(k => [k, 0])), outs = new Map(keys.map(k => [k, new Set()]));
      for (const e of fl.edges) { const a = key(e.from.split('@')[0]), b = key(e.to); if (a === b || other.has(e.to) || outs.get(a).has(b)) continue; outs.get(a).add(b); indeg.set(b, indeg.get(b) + 1); }
      const first = k => (k[0] === 'L' && compOf.size && !byId.has(k) ? Math.min(...[...scans[+k.slice(1)].members].map(id => pos.get(id))) : pos.get(k));
      const ready = keys.filter(k => !indeg.get(k)); order = [];
      while (ready.length) {
        ready.sort((a, b) => first(a) - first(b)); const k = ready.shift();
        if (!byId.has(k) && compOf.size && k[0] === 'L') { const L = scans[+k.slice(1)]; const mem = order0.filter(id => L.members.has(id)); L.order = mem; order.push(...mem); } else order.push(k);
        for (const b of outs.get(k)) { indeg.set(b, indeg.get(b) - 1); if (!indeg.get(b)) ready.push(b); }
      }
      if (loops.some(L => !L.scan)) var loopDelays = new Set(loops.filter(L => !L.scan).flatMap(L => [...L.members].filter(id => byId.get(id).type === 'delay')));
    } else if (fl.nodes.some(n => n.type === 'delay')) {
      const indeg = new Map(fl.nodes.map(n => [n.id, (ix.ins.get(n.id) || []).filter(Boolean).length])), q = fl.nodes.filter(n => !indeg.get(n.id)).map(n => n.id), full = [];
      while (q.length) { const id = q.shift(); full.push(id); for (const o of ix.outs.get(id) || []) { indeg.set(o.to, indeg.get(o.to) - 1); if (!indeg.get(o.to)) q.push(o.to); } }
      // Delays the plain order couldn't place are in a loop: they come first and hand over the previous state
      if (full.length < fl.nodes.length) { var loopDelays = new Set(fl.nodes.filter(n => n.type === 'delay' && !full.includes(n.id)).map(n => n.id)); order = order0; }
      else order = full;
    }
    if (conflicts.size) return { error: 'Fix the size conflicts first (marked in red).' };
    if (errors.size) { const [id, m] = [...errors][0]; return { error: m, node: id }; }
    const shapeOf = id => { const s = shapes.get(id); if (!s) return null; const v = s.map(d => C.evalDim(d, sym)); return v.every(x => Number.isFinite(x)) ? v : null; };
    const consumers = id => (ix.outs.get(id) || []).map(o => byId.get(o.to)).filter(Boolean);
    const notes = new Set();

    // ── positions: sizes that index the sequence (input dims other than the batch and the features) ──
    const posSyms = new Set();
    const GMAX = 8; // image sides simulated pixel by pixel; larger maps wrap around at this size
    const isImgShape = s => !!s && s.length === 4;
    for (const n of fl.nodes) if (n.type === 'input') {
      const d = C.parseShape(n.params.shape), emb = consumers(n.id).some(c => c.type === 'embed');
      if (d.length === 4 && !emb) continue; // an image: its positions are pixels
      (emb ? d.slice(1) : d.slice(1, -1)).forEach(x => posSyms.add(String(x)));
    }
    let Tdata = 1; for (const s of posSyms) { const v = C.evalDim(s, sym); if (v > Tdata) Tdata = v; }
    const TS = Math.max(1, Math.min(Tdata, TCAP));
    if (Tdata > TCAP) notes.add(`Sequences are simulated with their first ${TCAP} positions.`);
    const posT = id => { const s = shapes.get(id); return s && s.length >= 2 && posSyms.has(String(s[s.length - 2])) ? TS : 1; };
    let Dh = 0; for (const n of fl.nodes) if (n.type === 'heads') { const s = shapeOf(n.id); if (s) Dh = Math.max(Dh, s[s.length - 1]); }
    let Pimg = 1; for (const n of fl.nodes) { const s = shapeOf(n.id); if (isImgShape(s)) Pimg = Math.max(Pimg, Math.min(s[2], GMAX) * Math.min(s[3], GMAX)); }
    // image models carry more values, so a classifier head after the convolutions (one position) still has thousands of samples
    // an image flattened into a sequence (ViT's patches) is a sequence as long as the map, plus any tokens put in front
    // a parameter that is a matrix (leading sizes of 1 aside, not a position table): Mamba's d_inner × d_state A_log
    function innerParam(n) { if (n.type !== 'param' || n.params.init === 'sincos2d') return false; const sh = (shapeOf(n.id) || []).slice(); while (sh.length && sh[0] === 1) sh.shift(); return sh.length >= 2 && sh[sh.length - 1] > 1; }
    let Tseq = 1; for (const n of fl.nodes) if (n.type === 'func' && /(^|\.)flatten$/.test(n.params.fn) && /^\s*(2|-2)\s*$/.test(String(n.params.extra || '').split(',')[0])) { const s = shapeOf((ix.ins.get(n.id) || [])[0]); if (isImgShape(s)) Tseq = Math.max(Tseq, Math.min(s[2] * s[3], SEQCAP) + 1); }
    let NS = Math.max(256, Math.floor((Pimg > 1 ? 2 * TOTAL : TOTAL) / Math.max(TS, Pimg, Tseq))); if (Dh > 0) NS = Math.max(Dh, Math.floor(NS / Dh) * Dh);
    // tensors with an inner axis (a parameter like Mamba's d_inner × d_state A_log) carry KMAX entries per column:
    // fewer columns keep them within the same budget
    if (fl.nodes.some(innerParam)) NS = Math.max(128, Math.floor(NS / 4));

    // ── random numbers and tensors ──
    const rand = mulberry(0x5eed + 7919 * (opts._seed || 0)) /* a recurrence's repeated steps each draw new inputs */; let spare = null;
    const gauss = () => { if (spare !== null) { const v = spare; spare = null; return v; } let u = 0; while (u === 0) u = rand(); const r = Math.sqrt(-2 * Math.log(u)), a = 2 * Math.PI * rand(); spare = r * Math.sin(a); return r * Math.cos(a); };
    // NS standard normal values, one from each of NS equal-probability slices, in random order: the samples have
    // the normal distribution's shape almost exactly, so averages of nonlinear functions carry little sampling noise
    function strata() {
      if (!strata.base || strata.base.length !== NS) { strata.base = new Float64Array(NS); for (let i = 0; i < NS; i++) strata.base[i] = probit((i + 0.5) / NS); }
      const z = Float64Array.from(strata.base);
      for (let i = NS - 1; i > 0; i--) { const j = (rand() * (i + 1)) | 0; const q = z[i]; z[i] = z[j]; z[j] = q; }
      return z;
    }
    const feat = (T, X) => ({ k: 'f', T, X });
    // independent N(0, v) values, rescaled to exactly mean 0 and variance v (less sampling noise)
    const iid = (T, v) => {
      const X = new Float32Array(T * NS); let m = 0, q = 0;
      for (let t = 0; t < T; t++) { const z = strata(); for (let s = 0; s < NS; s++) { X[t * NS + s] = z[s]; m += z[s]; } }
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
      const X = new Float32Array(T * NS), z = new Float64Array(T), Z = Array.from({ length: T }, strata);
      for (let s = 0; s < NS; s++) { for (let t = 0; t < T; t++) z[t] = Z[t][s]; for (let t = 0; t < T; t++) { let v = 0; for (let u = 0; u <= t; u++) v += L[t * T + u] * z[u]; X[t * NS + s] = v; } }
      // rescale each position to its exact target variance (correlated positions don't average sampling noise away)
      for (let t = 0; t < T; t++) { let q = 0; for (let s = 0; s < NS; s++) q += X[t * NS + s] * X[t * NS + s]; const k = q > 0 ? Math.sqrt(K[t * T + t] / (q / NS)) : 0; for (let s = 0; s < NS; s++) X[t * NS + s] *= k; }
      return feat(T, X);
    };
    const meanAll = X => { let s = 0; for (let i = 0; i < X.length; i++) s += X[i]; return s / X.length; };
    const varAll = X => { const m = meanAll(X); let s = 0; for (let i = 0; i < X.length; i++) { const d = X[i] - m; s += d * d; } return s / X.length; };
    const mapX = (A, f) => { const X = new Float32Array(A.X.length); for (let i = 0; i < X.length; i++) X[i] = f(A.X[i], i); return Object.assign({}, A, { X }); };
    const like = (A, X) => Object.assign({}, A, { X });

    // ── images: pixels as positions ──
    // rep: the actual size (h × w), the simulated grid (gh × gw), whether it wraps around (torus), and how many grid
    // steps one actual pixel is (dh, dw; more than 1 after a pooling on a wrapped map, which keeps every grid point)
    const imgRep = (h, w) => { const torus = h > GMAX || w > GMAX; return { h, w, gh: Math.min(h, GMAX), gw: Math.min(w, GMAX), torus, dh: 1, dw: 1 }; };
    const exactRep = (h, w) => ({ h, w, gh: h, gw: w, torus: false, dh: 1, dw: 1 });
    const wrapd = (d, g) => { d = ((d % g) + g) % g; return d > g / 2 ? d - g : d; };
    const near = g => Math.max(0, Math.floor(g / 2) - 1); // grid displacements with their own statistics; beyond: the far value
    // a wrapped map's second moments by grid displacement, and the far value (what pixels far apart share)
    function stat(A, X) {
      const R = A.img, gh = R.gh, gw = R.gw, P = gh * gw, M = moments({ T: P, X: X || A.X });
      const S = new Float64Array(P), cnt = new Float64Array(P);
      for (let p = 0; p < P; p++) for (let q = 0; q < P; q++) { const d = ((Math.floor(q / gw) - Math.floor(p / gw) + gh) % gh) * gw + ((q % gw) - (p % gw) + gw) % gw; S[d] += M[p * P + q]; cnt[d]++; }
      for (let d = 0; d < P; d++) S[d] /= cnt[d] || 1;
      const rh = near(gh), rw = near(gw); let fs = 0, fc = 0;
      for (let d = 0; d < P; d++) { const a = Math.abs(wrapd(Math.floor(d / gw), gh)), b = Math.abs(wrapd(d % gw, gw)); if (a > rh || b > rw) { fs += S[d]; fc++; } }
      return { S, far: fc ? fs / fc : S[P - 1], rh, rw, gh, gw, R };
    }
    // E[x_a x_b] for pixels an actual displacement (Dh, Dw) apart
    const kat = (st, Dh, Dw) => {
      const th = Math.round(Dh * st.R.dh), tw = Math.round(Dw * st.R.dw);
      if (Math.abs(th) > st.rh || Math.abs(tw) > st.rw) return st.far;
      return st.S[(((th % st.gh) + st.gh) % st.gh) * st.gw + (((tw % st.gw) + st.gw) % st.gw)];
    };
    // a stationary covariance on a wrapped grid, from a function of the actual displacement
    const statCov = (R, fn) => {
      const P = R.gh * R.gw, K = new Float64Array(P * P);
      for (let p = 0; p < P; p++) for (let q = p; q < P; q++) {
        const v = fn(wrapd(Math.floor(q / R.gw) - Math.floor(p / R.gw), R.gh) / R.dh, wrapd((q % R.gw) - (p % R.gw), R.gw) / R.dw);
        K[p * P + q] = K[q * P + p] = v;
      }
      return K;
    };
    const valid = (o, t, s, pad, n) => { const a = o * s - pad + t; return a >= 0 && a < n; };
    // taps of a window that land inside the map, for output index o along one axis
    const taps = (o, k, s, pad, n) => { let c = 0; for (let t = 0; t < k; t++) if (valid(o, t, s, pad, n)) c++; return c; };
    // average share of a window's taps inside the map (the padding's effect, averaged over the output)
    const tapShare = (nOut, k, s, pad, nIn) => { let c = 0; for (let o = 0; o < nOut; o++) c += taps(o, k, s, pad, nIn); return c / (nOut * k); };
    // how many (output, tap) pairs read each input pixel, on average along one axis
    const readShare = (nIn, nOut, k, s, pad) => { let c = 0; for (let o = 0; o < nOut; o++) c += taps(o, k, s, pad, nIn); return c / nIn; };
    // a wrapped map with dil 1 cut to its actual size (≤ the grid), to line up with a map simulated exactly
    const toExact = A => {
      const R = A.img; if (!R || !R.torus || R.h > R.gh || R.w > R.gw || R.dh !== 1 || R.dw !== 1) return A;
      const X = new Float32Array(R.h * R.w * NS);
      for (let i = 0; i < R.h; i++) for (let j = 0; j < R.w; j++) X.set(A.X.subarray((i * R.gw + j) * NS, (i * R.gw + j + 1) * NS), (i * R.w + j) * NS);
      return Object.assign({}, A, { T: R.h * R.w, X, img: exactRep(R.h, R.w) });
    };
    const alignImgs = list => {
      if (!list.every(v => v && v.img)) return list;
      const ex = list.some(v => !v.img.torus) ? list.map(toExact) : list;
      if (ex.some(v => v.T !== ex[0].T)) return null;
      if (ex.some(v => v.img.dh !== ex[0].img.dh || v.img.dw !== ex[0].img.dw)) notes.add('Feature maps with different pooling histories are added as if aligned pixel for pixel.');
      return ex;
    };
    const permuted = () => { const perm = Int32Array.from({ length: NS }, (_, i) => i); for (let i = NS - 1; i > 0; i--) { const j = (rand() * (i + 1)) | 0; const q = perm[i]; perm[i] = perm[j]; perm[j] = q; } return perm; };
    const names = linearNames(fl, order, byId);
    // a tied layer uses the weights (so the init) of the layer it shares with
    const ownVar = o => {
      const q = o.params;
      if (o.type === 'embed') { const nm = C.evalDim(q.num, sym), dm = C.evalDim(q.dim, sym); return (q.init || 'default') === 'default' ? { v: 1, text: '1' } : { v: weightVar(q, dm, nm), text: weightVarText(q, dm, nm) }; }
      const fi = C.evalDim(q.in, sym), fo = C.evalDim(q.out, sym); return { v: weightVar(q, fi, fo), text: weightVarText(q, fi, fo) };
    };
    const tiedVar = n => { const o = C.tieOwner(fl, n, byId); if (!o) return null; const r = ownVar(o); return { v: r.v, text: `${r.text}, tied to ${o.params.title || names.get(o.id) || o.params.name || o.type}` }; };
    const val = new Map(), tape = new Map(), layers = [];
    // a constant, possibly written with sizes (d ** 0.5), as a real number
    const numConst = expr => {
      const c = Number(expr); if (Number.isFinite(c)) return c;
      const r = String(expr).replace(/[A-Za-z_]\w*/g, t => { const v = C.evalDim(t, sym); return v != null ? '(' + v + ')' : t; });
      if (/^[\d.\s+\-*/()eE]+$/.test(r)) { try { const v = Function('"use strict"; return (' + r + ');')(); if (Number.isFinite(v)) return v; } catch (e) { /* not a plain expression */ } }
      return C.evalDim(expr, sym);
    };
    const unsupported = (n, what) => ({ error: `Can't predict through ${what || n.params.fn || n.type}: it isn't one of the operations the variance plot knows.`, node: n.id });
    // ── scan loops: one position at a time, each Delay handing on what its input was one position earlier ──
    const scanOf = new Map(); scans.forEach(L => L.members.forEach(id => scanOf.set(id, L)));
    const BIN = { '+': (x, y) => x + y, '-': (x, y) => x - y, '*': (x, y) => x * y, '/': (x, y) => x / y };
    const elemFn = n => { const p = n.params; if (n.type === 'act') return p.fn === 'leaky_relu' ? [x => (x > 0 ? x : (+p.slope || 0.01) * x), x => (x > 0 ? 1 : (+p.slope || 0.01))] : ACT[p.fn]; const b = String(p.fn).replace(/^.*\./, ''); return ACT[/^(unsqueeze|squeeze|contiguous)$/.test(b) ? 'identity' : b]; };
    const constOf = p => { let c = numConst(p.const); return c; };
    function runScan(L) {
      const mem = L.order; let T = 1, K = 1;
      for (const id of mem) for (const s of ix.ins.get(id) || []) if (s && !L.members.has(s)) { const v = val.get(s); if (!v) return { error: 'A layer has an unconnected input.', node: id }; if (v.k !== 'f') return unsupported(byId.get(id), 'a loop over attention scores'); T = Math.max(T, v.T); if (v.K > 1) { if (K > 1 && v.K !== K) return unsupported(byId.get(id), 'a loop over tensors with different inner sizes'); K = v.K; } }
      // a step's slice: NS columns, each with K entries along an inner axis (Mamba's state); plain inputs repeat along it
      const W = NS * K, X = new Map(mem.map(id => [id, new Float32Array(T * W)])), rep = new Map();
      const sl = (s, t) => {
        if (X.has(s)) return X.get(s).subarray(t * W, (t + 1) * W);
        const v = val.get(s), u = v.T === 1 ? 0 : t;
        if ((v.K || 1) === K) return v.X.subarray(u * W, (u + 1) * W);
        const key = s + '|' + u; if (!rep.has(key)) { const Y = new Float32Array(W); for (let i = 0; i < NS; i++) Y.fill(v.X[u * NS + i], i * K, (i + 1) * K); rep.set(key, Y); } return rep.get(key);
      };
      for (const id of mem) { const n = byId.get(id); if (n.type === 'binop' && !BIN[n.params.op || '+']) return unsupported(n, `the operator ${n.params.op}`); if ((n.type === 'act' || n.type === 'func') && !elemFn(n)) return unsupported(n); if (n.type === 'binop' && (ix.ins.get(id) || []).length < 2 && !Number.isFinite(constOf(n.params))) return unsupported(n, `the constant "${n.params.const}"`); }
      for (let t = 0; t < T; t++) for (const id of mem) {
        const n = byId.get(id), p = n.params, o = X.get(id).subarray(t * W, (t + 1) * W), src = ix.ins.get(id) || [];
        if (n.type === 'delay') { if (t > 0) o.set(sl(src[0], t - 1)); continue; }
        const a = sl(src[0], t);
        if (n.type === 'binop') { const f = BIN[p.op || '+']; if (src[1]) { const b = sl(src[1], t); for (let i = 0; i < W; i++) o[i] = f(a[i], b[i]); } else { const c = constOf(p); for (let i = 0; i < W; i++) o[i] = p.rev ? f(c, a[i]) : f(a[i], c); } }
        else { const f = elemFn(n)[0]; for (let i = 0; i < W; i++) o[i] = f(a[i]); }
      }
      for (const id of mem) { const v = Object.assign(feat(T, X.get(id)), { shape: shapeOf(id) }); if (K > 1) v.K = K; val.set(id, v); }
      L.T = T; L.K = K; return null;
    }

    for (const id of order) {
      if (scanOf.has(id)) { const L = scanOf.get(id); if (L.order[0] === id) { const r = runScan(L); if (r) return r; } continue; }
      const n = byId.get(id), p = n.params, shape = shapeOf(id);
      const ins = (ix.ins.get(id) || []).map(s => val.get(s));
      const A = ins[0];
      const isLoopDelay = n.type === 'delay' && typeof loopDelays !== 'undefined' && loopDelays.has(id);
      const takesPositions = n.type === 'func' && (p.fn === 'take_positions');
      if (n.type !== 'input' && n.type !== 'embed' && !isLoopDelay && ins.some((v, j) => !v || (v.k === 'idx' && !(takesPositions && j === 1)))) {
        if (ins.some(v => v && v.k === 'idx')) return { error: 'Token indices can only go into an Embedding.', node: id };
        return { error: 'A layer has an unconnected input.', node: id };
      }
      let out = null;
      switch (n.type) {
        case 'input': {
          // positions for take_positions (I-JEPA's masks): only how many there are matters
          if (consumers(id).length && (ix.outs.get(id) || []).every(o => { const c = byId.get(o.to); return c && c.type === 'func' && c.params.fn === 'take_positions' && o.port === 1; })) { out = { k: 'idx', positions: true, K: shape ? shape[shape.length - 1] : 1 }; break; }
          const embs = consumers(id).filter(c => c.type === 'embed');
          if (!embs.length && isImgShape(shape)) { const R = imgRep(shape[2], shape[3]); out = iid(R.gh * R.gw, 1); out.img = R; break; }
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
          if (A.img && A.T > 1) return unsupported(n, 'a Linear on an image (flatten it first)');
          const fin = A.shape ? A.shape[A.shape.length - 1] : C.evalDim(p.in, sym), fout = C.evalDim(p.out, sym);
          const own = tiedVar(n), vw = own ? own.v : weightVar(p, fin, fout), vb = biasVar(p, fin), M = moments(A);
          if (opts._persist) {
            // in a recurrence, a unit's bias and what its weights make of the input's average are the same at every
            // step: drawn once per unit (sample column) and kept; the rest is drawn fresh at each step
            const mu = meanAll(A.X), pv = Math.max(0, vb + fin * vw * mu * mu);
            for (let i = 0; i < M.length; i++) M[i] = fin * vw * (M[i] - mu * mu);
            out = withCov(A.T, M);
            let z = opts._persist.get(id); if (!z || z.length !== NS) { z = strata(); opts._persist.set(id, z); }
            const sp = Math.sqrt(pv); for (let i = 0; i < out.X.length; i++) out.X[i] += sp * z[i % NS];
          } else {
            for (let i = 0; i < M.length; i++) M[i] = fin * vw * M[i] + vb;
            out = withCov(A.T, M);
          }
          if (p.bias && p.bias_init === 'dt') {
            // Mamba's time-step bias: one value per unit, softplus⁻¹ of a log-uniform step in [0.001, 0.1]
            let b = opts._persist && opts._persist.get(id + '|b');
            if (!b || b.length !== NS) { const pm = permuted(); b = new Float32Array(NS); for (let s2 = 0; s2 < NS; s2++) { const dt = Math.max(1e-4, 0.001 * Math.pow(100, (pm[s2] + 0.5) / NS)); b[s2] = dt + Math.log(-Math.expm1(-dt)); } if (opts._persist) opts._persist.set(id + '|b', b); }
            for (let i = 0; i < out.X.length; i++) out.X[i] += b[i % NS];
          }
          tape.set(id, { fout, vw });
          layers.push({ id, name: names.get(id), title: p.title || '', init: p.init || 'default', wvar: vw, wvarText: own ? own.text : weightVarText(p, fin, fout), bias: !!p.bias, fin, fout, fwdIn: varAll(A.X), fwdOut: varAll(out.X), bwdIn: null, bwdOut: null });
          break;
        }
        case 'norm': {
          const def = p.kind === 'RMSNorm' ? 1.1920928955078125e-7 : 1e-5, eps = p.eps !== '' && p.eps != null && isFinite(+p.eps) ? +p.eps : def;
          // BatchNorm2d normalizes each channel over the batch and the pixels: a sample column is one channel, so its own
          // offset (what its weights make of the input's mean) goes, and the scale is the spread within a channel
          if (p.kind === 'BatchNorm2d' && A.img && A.T >= 4) {
            const T = A.T, X = new Float32Array(A.X.length), cm = new Float64Array(NS); let q = 0;
            for (let t2 = 0; t2 < T; t2++) for (let s2 = 0; s2 < NS; s2++) cm[s2] += A.X[t2 * NS + s2] / T;
            // a column's mean over its pixels is the channel's offset (removed) plus this example's own share, worth
            // about (within-column variance) / pixels, which BatchNorm keeps: the mean is removed and that share redrawn
            let w = 0;
            for (let i = 0; i < A.X.length; i++) w += (A.X[i] - cm[i % NS]) ** 2; w /= A.X.length * (1 - 1 / T);
            const own = iid(1, w / T).X;
            for (let i = 0; i < X.length; i++) { const v = A.X[i] - cm[i % NS] + own[i % NS]; X[i] = v; q += v * v; }
            const sc = 1 / Math.sqrt(q / X.length + eps); for (let i = 0; i < X.length; i++) X[i] *= sc;
            out = feat(T, X); out.img = A.img; tape.set(id, { bn2d: sc });
            break;
          }
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
          out = feat(T, X); if (A.img) out.img = A.img; tape.set(id, { sets, scale, rms });
          break;
        }
        case 'act': {
          if (p.fn === 'softmax') { out = softmaxFwd(id, A, shape, p.dim); if (out.error) return out; break; }
          const sl = +(p.slope ?? 0.01), f = p.fn === 'leaky_relu' ? [x => (x > 0 ? x : sl * x), x => (x > 0 ? 1 : sl)] : ACT[p.fn]; if (!f) return unsupported(n, p.fn);
          out = mapX(A, f[0]); tape.set(id, { d: f[1] });
          break;
        }
        case 'dropout': out = A; notes.add('Dropout is treated as off (eval mode).'); break;
        case 'embed': {
          const nm = C.evalDim(p.num, sym), dm = C.evalDim(p.dim, sym);
          const own = tiedVar(n), v = own ? own.v : (p.init || 'default') === 'default' ? 1 : weightVar(p, dm, nm), T = p.positions && A && A.k === 'f' && !A.img ? A.T : posT(id);
          if (p.positions || T === 1 || !A || !A.ids) { out = iid(T, v); break; } // each position has its own row
          // positions holding the same token read the same row (this matters for small tables, such as BERT's two token types)
          const rows = new Map(), X = new Float32Array(T * NS);
          for (let t = 0; t < T; t++) { const k = A.ids[t % A.ids.length]; if (!rows.has(k)) rows.set(k, iid(1, v).X); X.set(rows.get(k), t * NS); }
          out = feat(T, X);
          break;
        }
        case 'heads': case 'merge': out = A; break;
        // standard Gaussian noise, independent of everything (no gradient flows back through it)
        case 'noise': { if (A.k !== 'f') return unsupported(n, 'noise shaped like attention scores'); out = iid(A.T, 1); if (A.img) out.img = A.img; break; }
        // a learned tensor as initialized: the same at every position; a column is one of its entries
        case 'param': {
          // a parameter with a last size N beyond its first (Mamba's A_log, d_inner × d_state) carries K = min(N, KMAX)
          // of its N entries in every column (an inner axis); a vector, one entry per column
          const sh = shape || [1], N = sh[sh.length - 1] || 1, K = innerParam(n) ? Math.min(N, KMAX) : 1, X = new Float32Array(NS * K);
          if (p.init === 'sincos2d' && sh.length >= 2) {
            // I-JEPA's fixed table: a column is one feature, at every one of the N positions (row-major on a square grid)
            const Np = Math.min(sh[sh.length - 2], SEQCAP), D = sh[sh.length - 1], gq = Math.round(Math.sqrt(sh[sh.length - 2])), q = Math.max(1, Math.floor(D / 4)), Y = new Float32Array(Np * NS);
            for (let t2 = 0; t2 < Np; t2++) for (let s2 = 0; s2 < NS; s2++) {
              const j = (s2 * 7919) % D, part = Math.floor(j / q), w = Math.pow(10000, -(j % q) / q), pos = part < 2 ? t2 % gq : Math.floor(t2 / gq);
              Y[t2 * NS + s2] = part % 2 ? Math.cos(pos * w) : Math.sin(pos * w);
            }
            out = feat(Np, Y); out.param = true; break;
          }
          if (p.init === 'ones') X.fill(1);
          else if (p.init === 'normal') { const z = strata(), sd = numConst(p.init_arg || '0.02'); for (let i = 0; i < X.length; i++) X[i] = sd * z[i % NS]; }
          else if (p.init === 's4d') { const pm = permuted(); for (let s2 = 0; s2 < NS; s2++) for (let k = 0; k < K; k++) X[s2 * K + k] = Math.log(1 + Math.floor(((K > 1 ? k : 0) + (pm[s2] + 0.5) / NS) * N / K)); } // log n, n = 1 … N, spread over the K
          out = feat(1, X); if (K > 1) out.K = K; out.param = true; break;
        }
        // a causal depthwise convolution along the positions: each channel (column) has its own k taps and bias,
        // PyTorch's default init U(±1/√k); out_t = Σ_j w_j · x_(t−k+1+j) + b, zeros before the first position
        case 'conv1d': {
          if (A.k !== 'f') return unsupported(n, 'a convolution over attention scores');
          const k = Math.max(1, C.evalDim(p.k, sym) || 1), bound = 1 / Math.sqrt(k), T = A.T, X = new Float32Array(T * NS);
          let w = opts._persist && opts._persist.get(id);
          if (!w || w.length !== (k + 1) * NS) { w = new Float32Array((k + 1) * NS); for (let j = 0; j <= k; j++) { const pm = permuted(); for (let s2 = 0; s2 < NS; s2++) w[j * NS + s2] = j === k && !p.bias ? 0 : bound * (2 * (pm[s2] + 0.5) / NS - 1); } if (opts._persist) opts._persist.set(id, w); }
          for (let t = 0; t < T; t++) for (let s2 = 0; s2 < NS; s2++) {
            let v = w[k * NS + s2]; for (let j = 0; j < k; j++) { const u = t - (k - 1) + j; if (u >= 0) v += w[j * NS + s2] * A.X[u * NS + s2]; }
            X[t * NS + s2] = v;
          }
          out = feat(T, X); tape.set(id, { c1: { w, k } });
          layers.push({ id, name: names.get(id), title: p.title || '', kind: 'conv', k, init: 'default', wvar: 1 / (3 * k), wvarText: `1/(3·${k})`, bias: !!p.bias, fin: k, fout: k, fwdIn: varAll(A.X), fwdOut: varAll(X), bwdIn: null, bwdOut: null });
          break;
        }
        // a learned token in front of the sequence (zeros at the start, as initialized)
        case 'token': {
          if (A.k !== 'f') return unsupported(n, 'a token in front of something that is not a sequence');
          const k = Math.max(1, C.evalDim(p.n, sym) || 1), X = new Float32Array((A.T + k) * NS); X.set(A.X, k * NS);
          out = feat(A.T + k, X); tape.set(id, { tok: k }); break;
        }
        // one position of the sequence
        case 'select': {
          if (A.k !== 'f') return unsupported(n, 'picking a position of something that is not a sequence');
          let i = C.evalDim(p.index, sym); if (!Number.isFinite(i)) i = 0; if (i < 0) i += A.T; i = Math.max(0, Math.min(A.T - 1, i));
          out = feat(1, A.X.slice(i * NS, (i + 1) * NS)); tape.set(id, { sel: i, T: A.T }); break;
        }
        case 'delay': if (typeof loopDelays !== 'undefined' && loopDelays.has(id)) {
          // the previous step's state: given (settled), or zeros on the first repeat
          const st = opts._state && opts._state.get(id), sh = shapeOf(id), T = sh ? posT(id) : 1;
          out = feat(T, st && st.length === T * NS ? Float32Array.from(st) : new Float32Array(T * NS)); break;
        } else { // one position later; zeros at the first
          const X = new Float32Array(A.X.length); X.set(A.X.subarray(0, (A.T - 1) * NS), NS); out = Object.assign({}, A, { X }); break;
        }
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
            let c = numConst(p.const);
            if (!Number.isFinite(c)) return unsupported(n, `the constant "${p.const}"`);
            out = mapX(A, p.rev ? x => f(c, x) : x => f(x, c)); tape.set(id, { c });
            break;
          }
          if (A.k !== B.k || (A.k === 's' && (A.T !== B.T || A.Hs !== B.Hs))) return unsupported(n, `${op} between attention scores and features`);
          if (A.k === 'f' && [A, B].some(v => v.K > 1 || v.bc === 'inner') && shape && shape.length >= 3 && shape[shape.length - 1] > 1) {
            // along an inner axis: each column carries K entries (one unit's, sharing its other values)
            const K = Math.max(...[A, B].map(v => v.K || 1)) > 1 ? Math.max(...[A, B].map(v => v.K || 1)) : Math.min(shape[shape.length - 1], KMAX);
            if ([A, B].some(v => v.K > 1 && v.K !== K)) return unsupported(n, `${op} of tensors with different inner sizes`);
            const T = Math.max(A.T, B.T), X = new Float32Array(T * NS * K), STR = Math.max(1, Math.floor(NS / K));
            const get = v => (v.K > 1 ? (t, s2, k) => v.X[((v.T === 1 ? 0 : t) * NS + s2) * K + k] : v.bc === 'inner' ? (t, s2, k) => v.X[(v.T === 1 ? 0 : t) * NS + ((s2 + k * STR) % NS)] : (t, s2) => v.X[(v.T === 1 ? 0 : t) * NS + s2]);
            const ga = get(A), gb = get(B);
            for (let t2 = 0; t2 < T; t2++) for (let s2 = 0; s2 < NS; s2++) for (let k = 0; k < K; k++) X[(t2 * NS + s2) * K + k] = f(ga(t2, s2, k), gb(t2, s2, k));
            out = feat(T, X); out.K = K; tape.set(id, { kb: { K, STR, A, B } });
            break;
          }
          if (A.img && B.img) { const al = alignImgs([A, B]); if (!al) return unsupported(n, `${op} of feature maps of different sizes`); ins[0] = al[0]; ins[1] = al[1]; }
          const imgOut = A.img && B.img ? ins[0].img : null;
          if (A.k === 's') { out = like(A, A.X.map((x, i) => f(x, B.X[i]))); break; }
          const A2 = ins[0], B2 = ins[1], T = Math.max(A2.T, B2.T), X = new Float32Array(T * NS);
          for (let t = 0; t < T; t++) { const a = (A2.T === 1 ? 0 : t) * NS, b = (B2.T === 1 ? 0 : t) * NS, o = t * NS; for (let s = 0; s < NS; s++) X[o + s] = f(A2.X[a + s], B2.X[b + s]); }
          out = feat(T, X); if (imgOut) out.img = imgOut; else if (A2.img && A2.T === T) out.img = A2.img; else if (B2.img && B2.T === T) out.img = B2.img;
          tape.set(id, Object.assign(tape.get(id) || {}, { aligned: [A2, B2] }));
          break;
        }
        case 'func': {
          const fn = String(p.init || p.fn || ''), base = fn.replace(/\(.*$/, '').replace(/^.*\./, '').toLowerCase();
          // flattening an image: each flattened coordinate is one channel at one pixel
          // positions picked from a sequence (I-JEPA's masks): K of its T, spread over it (which ones doesn't matter
          // to the statistics); a table without positions (T = 1) is the same everywhere
          if (fn === 'take_positions' && ins.length === 2 && A.k === 'f') {
            const K = Math.max(1, (shape || [])[1] || (ins[1] && ins[1].K) || 1), T0 = A.T, at = Array.from({ length: K }, (_, k) => (T0 === 1 ? 0 : Math.min(T0 - 1, Math.floor((k + 0.5) * T0 / K))));
            const X = new Float32Array(K * NS); at.forEach((u, k) => X.set(A.X.subarray(u * NS, (u + 1) * NS), k * NS));
            out = feat(K, X); tape.set(id, { pick: at, T0 }); break;
          }
          if (fn === 'last_positions' && ins.length === 2 && A.k === 'f') {
            const K = Math.min(A.T, ins[1].T || 1), at = Array.from({ length: K }, (_, k) => A.T - K + k);
            out = feat(K, A.X.slice((A.T - K) * NS)); tape.set(id, { pick: at, T0: A.T }); break;
          }
          // the fixed sine/cosine positions of "Attention Is All You Need": a column is one feature (2i or 2i + 1)
          // the table alone (added by a + after it): the same values for every example, no gradient to x
          if (fn === 'sinusoidal_pe' && ins.length === 1 && A.k === 'f') {
            const D = (A.shape || [])[A.shape.length - 1] || NS, X = new Float32Array(A.T * NS);
            for (let t2 = 0; t2 < A.T; t2++) for (let s2 = 0; s2 < NS; s2++) { const j = (s2 * 7919) % D, f = Math.pow(10000, -(j - (j % 2)) / D); X[t2 * NS + s2] = j % 2 ? Math.cos(t2 * f) : Math.sin(t2 * f); }
            out = feat(A.T, X); out.param = true; tape.set(id, { none: true }); break;
          }
          if (fn === 'sinusoidal_positions' && ins.length === 1 && A.k === 'f') {
            const D = (A.shape || [])[A.shape.length - 1] || NS, X = new Float32Array(A.X.length);
            for (let t2 = 0; t2 < A.T; t2++) for (let s2 = 0; s2 < NS; s2++) { const j = (s2 * 7919) % D, f = Math.pow(10000, -(j - (j % 2)) / D); X[t2 * NS + s2] = A.X[t2 * NS + s2] + (j % 2 ? Math.cos(t2 * f) : Math.sin(t2 * f)); }
            out = feat(A.T, X); tape.set(id, { pass: true }); break;
          }
          // flattening an image's height and width (ViT's patches): a sequence of pixels, channels first (a transpose
          // usually follows); a wrapped map gives its simulated grid as the sequence
          if (base === 'flatten' && A && A.img && /^\s*(2|-2)\s*$/.test(String(p.extra || '').split(',')[0])) {
            const R = A.img;
            if (!R.torus || (R.dh === 1 && R.dw === 1 && R.h * R.w <= R.gh * R.gw)) { const E2 = toExact(A); out = { k: 'fT', T: E2.T, X: E2.X }; tape.set(id, { pass: true, exact: E2 !== A }); break; }
            // a wrapped map: every actual pixel, up to SEQCAP of them, row by row; pixels outside the simulated grid
            // are independent copies of the grid pixel they wrap onto (a convolution that feeds a sequence is simulated
            // pixel by pixel when it fits, so this is for maps that don't)
            const P = Math.min(R.h * R.w, SEQCAP), X = new Float32Array(P * NS), src = [];
            if (R.h * R.w > SEQCAP) notes.add(`A ${R.h} × ${R.w} map flattened into a sequence is simulated with its first ${SEQCAP} positions.`);
            const perms = new Map();
            for (let a = 0; a < P; a++) {
              const i = Math.floor(a / R.w), j = a % R.w, gi = Math.round(i * R.dh) % R.gh, gj = Math.round(j * R.dw) % R.gw, q = gi * R.gw + gj;
              const copy = Math.floor(Math.round(i * R.dh) / R.gh) * 1000 + Math.floor(Math.round(j * R.dw) / R.gw);
              const key = q + ':' + copy; if (!perms.has(key)) perms.set(key, copy ? permuted() : null);
              const pm = perms.get(key); for (let s2 = 0; s2 < NS; s2++) X[a * NS + s2] = A.X[q * NS + (pm ? pm[s2] : s2)];
              src.push([q, pm]);
            }
            out = { k: 'fT', T: P, X }; tape.set(id, { seqOf: { src, P1: A.T } });
            break;
          }
          if (base === 'flatten' && A && A.img) {
            const P = A.T, X = new Float32Array(NS), pick = new Int32Array(NS);
            for (let s2 = 0; s2 < NS; s2++) { pick[s2] = P > 1 ? Math.floor(rand() * P) : 0; X[s2] = A.X[pick[s2] * NS + s2]; }
            out = feat(1, X); tape.set(id, { flat: P });
            break;
          }
          if (/^(cat|concat|concatenate|stack)$/.test(base) && ins.length > 1 && ins.every(v => v && v.img)) { const al = alignImgs(ins); if (!al) return unsupported(n, fn + ' of feature maps of different sizes'); al.forEach((v, i) => { ins[i] = v; }); }
          // x.unsqueeze(-1): the same value all along a new last axis; x.unsqueeze(-2): its features become that axis
          // (Mamba: Δ ⊙ A over the state, B and C shared by every unit). Nothing moves until an op combines them.
          if (base === 'unsqueeze' && ins.length === 1 && A.k === 'f' && !(A.K > 1)) {
            const k0 = +String(p.extra).trim(); out = Object.assign({}, A, { bc: k0 === -1 ? 'last' : k0 === -2 ? 'inner' : undefined }); tape.set(id, { pass: true }); break;
          }
          if (base === 'sum' && ins.length === 1 && A.K > 1 && String(p.extra).trim() === '-1') {
            // the inner axis summed: its K entries stand for all N, so the sum is scaled by √(N/K)
            const K = A.K, sh = A.shape || [], N = sh[sh.length - 1] || K, sc = Math.sqrt(N / K), W = A.T * NS, X = new Float32Array(W);
            for (let i = 0; i < W; i++) { let v = 0; for (let k = 0; k < K; k++) v += A.X[i * K + k]; X[i] = sc * v; }
            out = feat(A.T, X); tape.set(id, { ksum: K }); break;
          }
          // summing over a size-K dimension (Mamba's Σ_n h·C over the state): K independent terms, simulated as
          // the column itself plus other columns (other units, the same distribution), scaled up to K terms
          if (/^(sum)$/.test(base) && ins.length === 1 && /^\s*-?\d+\s*$/.test(String(p.extra || ''))) {
            if (A.k !== 'f') return unsupported(n, 'a sum over attention scores');
            const sh = A.shape || [], kx = +p.extra, K = sh[kx < 0 ? sh.length + kx : kx] || 1, Kc = Math.min(K, 16), sc = Math.sqrt(K / Kc), X = new Float32Array(A.X.length);
            for (let c = 0; c < Kc; c++) { const pm = c ? permuted() : null; for (let t = 0; t < A.T; t++) for (let s2 = 0; s2 < NS; s2++) X[t * NS + s2] += sc * A.X[t * NS + (pm ? pm[s2] : s2)]; }
            out = feat(A.T, X); tape.set(id, { pass: true }); break;
          }
          if (/^(matmul|mm|bmm)$/.test(base) && ins.length === 2) { out = matmulFwd(id, ins[0], ins[1]); if (out.error) return out; break; }
          // joined along the positions (dim 1 of a sequence: I-JEPA's context tokens, then the target tokens)
          const catDim = (String(p.extra || '').match(/dim\s*=\s*(-?\d+)/) || [])[1], rk = (shape || []).length;
          if (/^(cat|concat|concatenate)$/.test(base) && ins.length > 1 && rk >= 3 && catDim != null && ((+catDim + rk) % rk) === rk - 2 && ins.every(v => v && v.k === 'f' && !v.img)) {
            const T = ins.reduce((t2, v) => t2 + v.T, 0), X = new Float32Array(T * NS); let o = 0;
            for (const v of ins) { X.set(v.X, o * NS); o += v.T; }
            out = feat(T, X); tape.set(id, { catPos: ins.map(v => v.T) }); break;
          }
          if (/^(cat|concat|concatenate|stack)$/.test(base) && ins.length > 1) {
            if (ins.some(v => v.k !== 'f')) return unsupported(n, fn + ' of attention scores');
            // each feature coordinate comes from one part, with probability equal to that part's share
            const w = ins.map(v => numel(v.shape || [1]) / Math.max(1, v.T)), tot = w.reduce((x, y) => x + y, 0);
            const T = Math.max(...ins.map(v => v.T)), X = new Float32Array(T * NS), pick = new Int32Array(NS);
            for (let s = 0; s < NS; s++) { let u = rand() * tot, k = 0; while (k < w.length - 1 && u >= w[k]) { u -= w[k]; k++; } pick[s] = k; }
            for (let t = 0; t < T; t++) for (let s = 0; s < NS; s++) { const v = ins[pick[s]]; X[t * NS + s] = v.X[(v.T === 1 ? 0 : t) * NS + s]; }
            out = feat(T, X); if (ins[0].img) out.img = ins[0].img; tape.set(id, { cat: true });
            break;
          }
          if (base === 'softmax') { const m = String(p.extra || '').match(/dim\s*=\s*(-?\d+)/); out = softmaxFwd(id, A, shape, m ? m[1] : -1); if (out.error) return out; break; }
          const act = { relu: 'relu', gelu: 'gelu', tanh: 'tanh', sigmoid: 'sigmoid', silu: 'silu', exp: 'exp', softplus: 'softplus', identity: 'identity', dropout: 'identity', contiguous: 'identity', reshape: 'identity', view: 'identity', flatten: 'identity', squeeze: 'identity', unsqueeze: 'identity' }[base];
          if (!act || ins.length !== 1) return unsupported(n, fn);
          out = mapX(A, ACT[act][0]); tape.set(id, { d: ACT[act][1] });
          break;
        }
        case 'conv': {
          if (!A.img || !isImgShape(A.shape)) return unsupported(n, 'a convolution on something that is not an image');
          if (p.transposed) {
            // ConvTranspose2d with stride = kernel and no padding (U-Net's 2×2 /2): every output pixel reads one input
            // pixel, through the kernel offset it sits at; pixels at different offsets use different weights, so they
            // are uncorrelated. PyTorch's default init takes fan_in from the weight's second dimension: C_out · k².
            const Cin = A.shape[1], Cout = shape[1], k = C.evalDim(p.k, sym), st0 = C.evalDim(p.stride, sym), pad = C.evalDim(p.pad, sym);
            if (k !== st0 || pad) return unsupported(n, 'a transposed convolution whose stride differs from its kernel (or with padding)');
            const fin = Cout * k * k, vw = weightVar(p, fin, Cin * k * k), vb = biasVar(p, fin), RI = A.img, RO = imgRep(shape[2], shape[3]), P2 = RO.gh * RO.gw, M = moments(A);
            const par = (g, R) => ((Math.round(Math.floor(g / k) * R) % 1e9));
            const src = P => { const gi = Math.floor(P / RO.gw), gj = P % RO.gw; return { p: (par(gi, RI.dh) % RI.gh) * RI.gw + (par(gj, RI.dw) % RI.gw), o: (gi % k) * k + (gj % k) }; };
            const sP = Array.from({ length: P2 }, (_, P) => src(P)), K = new Float64Array(P2 * P2);
            for (let P = 0; P < P2; P++) for (let Q = P; Q < P2; Q++) K[P * P2 + Q] = K[Q * P2 + P] = (sP[P].o === sP[Q].o ? Cin * vw * M[sP[P].p * A.T + sP[Q].p] : 0) + vb;
            out = withCov(P2, K); out.img = RO;
            tape.set(id, { convT: { RI, RO, k, f: Cout * vw, sP } });
            layers.push({ id, name: names.get(id), title: p.title || '', kind: 'conv', k, init: 'default', wvar: vw, wvarText: weightVarText(p, fin, Cin * k * k), bias: !!p.bias, fin, fout: Cin * k * k, fwdIn: varAll(A.X), fwdOut: varAll(out.X), bwdIn: null, bwdOut: null });
            break;
          }
          const Cin = A.shape[1], Cout = shape[1], k = C.evalDim(p.k, sym), st0 = C.evalDim(p.stride, sym), pad = C.evalDim(p.pad, sym);
          const fin = Cin * k * k, fout = Cout * k * k, vw = weightVar(p, fin, fout), vb = biasVar(p, fin);
          const toSeq = consumers(id).some(c => c.type === 'func' && /(^|\.)flatten$/.test(c.params.fn) && /^\s*(2|-2)\s*$/.test(String(c.params.extra || '').split(',')[0]));
          // a map that becomes a sequence (ViT's patches) is simulated pixel by pixel, so every patch is its own position
          const RI = A.img, RO = toSeq && shape[2] * shape[3] <= SEQCAP ? exactRep(shape[2], shape[3]) : imgRep(shape[2], shape[3]), P2 = RO.gh * RO.gw, K = new Float64Array(P2 * P2), g = Cin * vw;
          if (!RO.torus) {
            // pixel by pixel: the taps of P and Q at the same kernel offset, both inside the map
            const M = RI.torus ? null : moments(A), st = RI.torus ? stat(A) : null, P1 = RI.gh * RI.gw;
            for (let P = 0; P < P2; P++) for (let Q = P; Q < P2; Q++) {
              const pi = Math.floor(P / RO.gw), pj = P % RO.gw, qi = Math.floor(Q / RO.gw), qj = Q % RO.gw; let sum = 0;
              if (M) { for (let ti = 0; ti < k; ti++) { if (!valid(pi, ti, st0, pad, RI.h) || !valid(qi, ti, st0, pad, RI.h)) continue; for (let tj = 0; tj < k; tj++) { if (!valid(pj, tj, st0, pad, RI.w) || !valid(qj, tj, st0, pad, RI.w)) continue; const a = (pi * st0 - pad + ti) * RI.gw + (pj * st0 - pad + tj), b = (qi * st0 - pad + ti) * RI.gw + (qj * st0 - pad + tj); sum += M[a * P1 + b]; } } }
              else { let nh = 0, nw = 0; for (let t = 0; t < k; t++) { if (valid(pi, t, st0, pad, RI.h) && valid(qi, t, st0, pad, RI.h)) nh++; if (valid(pj, t, st0, pad, RI.w) && valid(qj, t, st0, pad, RI.w)) nw++; } sum = nh * nw * kat(st, (qi - pi) * st0, (qj - pj) * st0); }
              K[P * P2 + Q] = K[Q * P2 + P] = g * sum + vb;
            }
          } else {
            // a wrapped map: the same at every pixel, with the padding's share of the taps averaged in
            const st = stat(A);
            K.set(statCov(RO, (dh, dw) => g * k * k * kat(st, dh * st0, dw * st0)));
          }
          out = withCov(P2, K); out.img = RO;
          if (RO.torus) {
            // each value is an actual output pixel picked at random: near the edges fewer taps fall inside the map, so
            // its share of the full window's variance is (taps inside / k)² per axis; the bias is one value per channel
            const th = Array.from({ length: RO.h }, (_, o) => taps(o, k, st0, pad, RI.h) / k), tw = Array.from({ length: RO.w }, (_, o) => taps(o, k, st0, pad, RI.w) / k);
            const bias = vb > 0 ? iid(1, vb).X : null;
            for (let i = 0; i < out.X.length; i++) out.X[i] = out.X[i] * Math.sqrt(th[(rand() * RO.h) | 0] * tw[(rand() * RO.w) | 0]) + (bias ? bias[i % NS] : 0);
          }
          tape.set(id, { conv: { RI, RO, k, s: st0, pad, f: Cout * vw } });
          layers.push({ id, name: names.get(id), title: p.title || '', kind: 'conv', k, init: 'default', wvar: vw, wvarText: weightVarText(p, fin, fout), bias: !!p.bias, fin, fout, fwdIn: varAll(A.X), fwdOut: varAll(out.X), bwdIn: null, bwdOut: null });
          break;
        }
        case 'pool': {
          if (!A.img || !isImgShape(A.shape)) return unsupported(n, 'pooling on something that is not an image');
          const RI = A.img, X0 = A.X;
          if (p.kind === 'adaptive') {
            const z = shape[2], zw = shape[3];
            if (z === 1 && zw === 1) {
              const N = RI.h * RI.w, X = new Float32Array(NS);
              if (!RI.torus) { for (let q = 0; q < RI.gh * RI.gw; q++) for (let s2 = 0; s2 < NS; s2++) X[s2] += X0[q * NS + s2] / N; out = { k: 'f', T: 1, X, img: exactRep(1, 1) }; }
              else {
                // the average of a large map: what its pixels share, plus what is left of their own variation
                const st = stat(A), mu = meanAll(X0), far = st.far - mu * mu; let loc = 0;
                const Rh = Math.floor(st.rh / RI.dh), Rw = Math.floor(st.rw / RI.dw);
                for (let a = -Rh; a <= Rh; a++) for (let b = -Rw; b <= Rw; b++) loc += (kat(st, a, b) - mu * mu - far) * ((RI.h - Math.abs(a)) / RI.h) * ((RI.w - Math.abs(b)) / RI.w);
                const v = Math.max(0, far + loc / N), Z = iid(1, v).X; for (let s2 = 0; s2 < NS; s2++) X[s2] = mu + Z[s2];
                out = { k: 'f', T: 1, X, img: exactRep(1, 1) };
              }
              tape.set(id, { gap: N });
              break;
            }
            if (RI.torus) return unsupported(n, 'adaptive pooling of a large map to more than 1 × 1');
            // exact windows, as PyTorch: rows floor(i·n/z) … ceil((i+1)·n/z)
            const X = new Float32Array(z * zw * NS), win = [];
            for (let i = 0; i < z; i++) for (let j = 0; j < zw; j++) {
              const cells = []; for (let a = Math.floor(i * RI.h / z); a < Math.ceil((i + 1) * RI.h / z); a++) for (let b = Math.floor(j * RI.w / zw); b < Math.ceil((j + 1) * RI.w / zw); b++) cells.push(a * RI.gw + b);
              win.push(cells); for (const c of cells) for (let s2 = 0; s2 < NS; s2++) X[(i * zw + j) * NS + s2] += X0[c * NS + s2] / cells.length;
            }
            out = { k: 'f', T: z * zw, X, img: exactRep(z, zw) }; tape.set(id, { win });
            break;
          }
          const k = C.evalDim(p.k, sym), st0 = p.stride !== '' && p.stride != null ? C.evalDim(p.stride, sym) : k, pad = C.evalDim(p.pad || '0', sym), mx = p.kind !== 'avg';
          let RO, wins;
          if (!RI.torus) {
            RO = exactRep(shape[2], shape[3]); wins = [];
            for (let i = 0; i < RO.h; i++) for (let j = 0; j < RO.w; j++) { const c = []; for (let ti = 0; ti < k; ti++) for (let tj = 0; tj < k; tj++) if (valid(i, ti, st0, pad, RI.h) && valid(j, tj, st0, pad, RI.w)) c.push((i * st0 - pad + ti) * RI.gw + (j * st0 - pad + tj)); wins.push(c); }
          } else {
            // a wrapped map keeps every grid point (the window at each), now st0 steps per actual pixel
            RO = { h: shape[2], w: shape[3], gh: RI.gh, gw: RI.gw, torus: true, dh: RI.dh * st0, dw: RI.dw * st0 }; wins = [];
            for (let i = 0; i < RI.gh; i++) for (let j = 0; j < RI.gw; j++) { const c = []; for (let ti = 0; ti < k; ti++) for (let tj = 0; tj < k; tj++) c.push(((i + (ti - pad) * RI.dh) % RI.gh + RI.gh) % RI.gh * RI.gw + ((j + (tj - pad) * RI.dw) % RI.gw + RI.gw) % RI.gw); wins.push(c); }
            // which taps of a window fall inside the map, for each actual output row and column (the window is k × k, row-major)
            const inside = (n2, nIn) => Array.from({ length: n2 }, (_, o) => Array.from({ length: k }, (_, t2) => valid(o, t2, st0, pad, nIn)));
            var edgeH = inside(RO.h, RI.h), edgeW = inside(RO.w, RI.w);
            notes.add('Pooling on maps larger than 8 × 8 is simulated without striding; neighbouring outputs come out a little more alike than they are.');
          }
          const P2 = wins.length, X = new Float32Array(P2 * NS), arg = mx ? new Int32Array(P2 * NS) : null, use = RI.torus ? new Uint8Array(P2 * NS * k * k) : null;
          for (let q = 0; q < P2; q++) { const c = wins[q]; for (let s2 = 0; s2 < NS; s2++) {
            // on a wrapped map, the window of a random actual output pixel: taps in the padding don't count
            const eh = use && edgeH[(rand() * edgeH.length) | 0], ew = use && edgeW[(rand() * edgeW.length) | 0];
            const ok = j => !use || (eh[(j / k) | 0] && ew[j % k]);
            if (use) for (let j = 0; j < c.length; j++) use[(q * NS + s2) * k * k + j] = ok(j) ? 1 : 0;
            if (mx) { let b = -Infinity, at = -1; for (let j = 0; j < c.length; j++) { if (!ok(j)) continue; const v = X0[c[j] * NS + s2]; if (v > b) { b = v; at = c[j]; } } X[q * NS + s2] = at < 0 ? 0 : b; arg[q * NS + s2] = at; }
            else { let t = 0; for (let j = 0; j < c.length; j++) if (ok(j)) t += X0[c[j] * NS + s2]; X[q * NS + s2] = t / (k * k); }
          } }
          out = { k: 'f', T: P2, X, img: RO }; tape.set(id, { pool: { mx, arg, wins, k, use, scale: RI.torus ? 1 / Math.sqrt(st0 * st0) : 1 } });
          break;
        }
        case 'upsample': {
          if (!A.img || !isImgShape(A.shape)) return unsupported(n, 'upsampling something that is not an image');
          const f = Math.round(C.evalDim(p.scale, sym) || 2), RI = A.img;
          if (p.mode && p.mode !== 'nearest') notes.add('Bilinear upsampling is simulated as nearest.');
          let RO, src;
          if (RI.torus && RI.dh % f === 0 && RI.dw % f === 0) { RO = Object.assign({}, RI, { h: shape[2], w: shape[3], dh: RI.dh / f, dw: RI.dw / f }); src = null; }
          else {
            RO = imgRep(shape[2], shape[3]); src = new Int32Array(RO.gh * RO.gw);
            for (let i = 0; i < RO.gh; i++) for (let j = 0; j < RO.gw; j++) src[i * RO.gw + j] = (Math.floor(i / f) % RI.gh) * RI.gw + (Math.floor(j / f) % RI.gw);
          }
          const X = src ? new Float32Array(src.length * NS) : A.X;
          if (src) for (let q = 0; q < src.length; q++) X.set(A.X.subarray(src[q] * NS, (src[q] + 1) * NS), q * NS);
          out = { k: 'f', T: RO.gh * RO.gw, X, img: RO }; tape.set(id, { up: { f, src, RI, RO } });
          break;
        }
        default: return unsupported(n);
      }
      if (out.k !== 'idx') out = Object.assign({}, out, { shape });
      val.set(id, out);
    }
    if (noGrad.size) notes.add('Followers (EMA ties) get no gradients: their layers show forward values only.');
    if (TS > 1 && Dh > 0) notes.add(`Attention is computed on ${TS} positions with ${Math.max(1, Math.floor(NS / Math.max(1, Dh || NS)))} simulated heads.`);

    if (typeof loopDelays !== 'undefined' && opts._fwdOnly) {
      // the state for the next step, column for column (a column is one unit, whose fixed parts carry over); its
      // fluctuating part came from this step's inputs, which the next step draws afresh, so they are independent
      const nextState = new Map();
      for (const d of loopDelays) {
        const v = val.get((ix.ins.get(d) || [])[0]); if (!v || v.k !== 'f') return { error: 'A Delay in a loop needs a tensor coming back to it.', node: d };
        nextState.set(d, Float32Array.from(v.X));
      }
      return { nextState };
    }
    // ── backward: a standard Gaussian gradient enters at every output ──
    const grad = new Map();
    const addGrad = (id, G) => {
      if (!id || !G) return; const v = val.get(id); if (!v || v.k === 'idx') return;
      // a gradient worked out on a map cut to its actual size goes back into the wrapped grid it came from
      if (v.img && v.img.torus && G.length !== v.T * NS && G.length === v.img.h * v.img.w * NS) { const Y = new Float32Array(v.T * NS); for (let i = 0; i < v.img.h; i++) for (let j = 0; j < v.img.w; j++) Y.set(G.subarray((i * v.img.w + j) * NS, (i * v.img.w + j + 1) * NS), (i * v.img.gw + j) * NS); G = Y; }
      // a tensor broadcast over positions receives the sum of their gradients
      { const Wd = NS * (v.K > 1 ? v.K : 1); if (v.k !== 's' && v.T === 1 && G.length > Wd) { const S = new Float32Array(Wd); for (let i = 0; i < G.length; i++) S[i % Wd] += G[i]; G = S; } }
      const cur = grad.get(id);
      if (!cur) grad.set(id, G); else for (let i = 0; i < cur.length; i++) cur[i] += G[i];
    };
    const mapG = (G, f) => { const Y = new Float32Array(G.length); for (let i = 0; i < G.length; i++) Y[i] = f(G[i], i); return Y; };
    for (const n of fl.nodes) if (n.type === 'output' && val.get(n.id) && val.get(n.id).k !== 'idx') { const v = val.get(n.id); grad.set(n.id, iid(v.T, 1).X); }
    function runScanBack(L) {
      const mem = L.order, T = L.T, K = L.K || 1, W = NS * K, D = new Map(mem.map(id => [id, new Float32Array(T * W)])), extG = new Map();
      for (const id of mem) { const G = grad.get(id); if (G) { if (G.length === T * W) D.get(id).set(G); else for (let t = 0; t < T; t++) D.get(id).set(G.subarray(0, W), t * W); grad.delete(id); } }
      const xs = (s, t) => { const v = val.get(s), u = v.T === 1 ? 0 : t, w = NS * (v.K || 1); if (w === W) return v.X.subarray(u * W, (u + 1) * W); const Y = new Float32Array(W); for (let i = 0; i < NS; i++) Y.fill(v.X[u * NS + i], i * K, (i + 1) * K); return Y; };
      const push = (s, t, f) => {
        if (D.has(s)) { const d = D.get(s); for (let i = 0; i < W; i++) d[t * W + i] += f(i); return; }
        const v = val.get(s); if (!v || v.k === 'idx') return; const w = NS * (v.K || 1); if (!extG.has(s)) extG.set(s, new Float32Array(v.T * w));
        const d = extG.get(s), u = v.T === 1 ? 0 : t;
        if (w === W) for (let i = 0; i < W; i++) d[u * W + i] += f(i); else for (let i = 0; i < W; i++) d[u * NS + Math.floor(i / K)] += f(i);
      };
      for (let t = T - 1; t >= 0; t--) for (let j = mem.length - 1; j >= 0; j--) {
        const id = mem[j], n = byId.get(id), p = n.params, src = ix.ins.get(id) || [], g = D.get(id).subarray(t * W, (t + 1) * W);
        if (n.type === 'delay') { if (t > 0) push(src[0], t - 1, i => g[i]); continue; }
        const a = xs(src[0], t);
        if (n.type === 'binop') {
          const op = p.op || '+';
          if (src[1]) {
            const b = xs(src[1], t);
            if (op === '+') { push(src[0], t, i => g[i]); push(src[1], t, i => g[i]); }
            else if (op === '-') { push(src[0], t, i => g[i]); push(src[1], t, i => -g[i]); }
            else if (op === '*') { push(src[0], t, i => g[i] * b[i]); push(src[1], t, i => g[i] * a[i]); }
            else { push(src[0], t, i => g[i] / b[i]); push(src[1], t, i => -g[i] * a[i] / (b[i] * b[i])); }
          } else {
            const c = constOf(p);
            push(src[0], t, op === '+' ? i => g[i] : op === '-' ? i => (p.rev ? -g[i] : g[i]) : op === '*' ? i => g[i] * c : i => (p.rev ? -g[i] * c / (a[i] * a[i]) : g[i] / c));
          }
        } else { const d = elemFn(n)[1]; push(src[0], t, i => g[i] * d(a[i])); }
      }
      for (const [s, G] of extG) addGrad(s, G);
    }
    for (let k = order.length - 1; k >= 0; k--) {
      if (scanOf.has(order[k])) { const L = scanOf.get(order[k]); if (L.order[L.order.length - 1] === order[k]) runScanBack(L); continue; }
      if (noGrad.has(order[k])) { grad.delete(order[k]); continue; }
      const id = order[k], n = byId.get(id), p = n.params, G = grad.get(id); if (!G) continue;
      grad.delete(id);
      const src = ix.ins.get(id) || [], ins = src.map(s => val.get(s)), A = ins[0], me = val.get(id), t = tape.get(id);
      switch (n.type) {
        case 'output': case 'dropout': case 'heads': case 'merge': addGrad(src[0], G); break;
        case 'token': addGrad(src[0], G.slice(t.tok * NS)); break;
        case 'param': case 'noise': break;
        case 'conv1d': {
          const { w, k } = t.c1, T = me.T, dX = new Float32Array(T * NS);
          for (let tt = 0; tt < T; tt++) for (let s2 = 0; s2 < NS; s2++) { const g = G[tt * NS + s2]; if (!g) continue; for (let j = 0; j < k; j++) { const u = tt - (k - 1) + j; if (u >= 0) dX[u * NS + s2] += w[j * NS + s2] * g; } }
          const L = layers.find(l => l.id === id); L.bwdOut = varAll(G); L.bwdIn = varAll(dX);
          addGrad(src[0], dX); break;
        }
        case 'select': { const dX = new Float32Array(t.T * NS); dX.set(G, t.sel * NS); addGrad(src[0], dX); break; }
        case 'delay': { if (typeof loopDelays !== 'undefined' && loopDelays.has(id)) break; const dX = new Float32Array(G.length); dX.set(G.subarray(NS), 0); addGrad(src[0], dX); break; }
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
          if (t.bn2d) {
            // dX = s · (G − mean G − Y · mean(G·Y)), the means over the channel's whole batch. A column is one example:
            // what a gradient shares across its pixels comes from that example (e.g. global average pooling), and over
            // a large batch it averages out rather than being removed, so the means are taken over all columns
            const Y = me.X, dX = new Float32Array(G.length); let mg = 0, mgy = 0;
            for (let i = 0; i < G.length; i++) { mg += G[i]; mgy += G[i] * Y[i]; } mg /= G.length; mgy /= G.length;
            for (let i = 0; i < G.length; i++) dX[i] = t.bn2d * (G[i] - mg - Y[i] * mgy);
            addGrad(src[0], dX);
            break;
          }
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
        case 'conv': {
          if (t.convT) {
            // each input pixel gets the gradients of its k × k children, one per kernel offset (independent weights)
            const { RI, RO, k, f, sP } = t.convT, P1 = RI.gh * RI.gw, P2 = RO.gh * RO.gw, Gm = moments({ T: P2, X: G });
            const kids = Array.from({ length: P1 }, () => new Map());
            for (let P = 0; P < P2; P++) { const m = kids[sP[P].p]; if (!m.has(sP[P].o)) m.set(sP[P].o, P); }
            // grid pixels with no child on the output grid (a wrapped output covers part of the input) borrow a covered pixel's
            const has = kids.map(m => m.size > 0), cov = kids.findIndex(m => m.size > 0);
            const K = new Float64Array(P1 * P1);
            for (let a = 0; a < P1; a++) for (let b = a; b < P1; b++) {
              const ma = kids[has[a] ? a : cov], mb = kids[has[b] ? b : cov]; let v = 0;
              for (const [o, Pa] of ma) { const Pb = mb.get(o); if (Pb !== undefined) v += Gm[Pa * P2 + Pb]; }
              if ((!has[a] || !has[b]) && a !== b) v = 0;
              K[a * P1 + b] = K[b * P1 + a] = f * v * (k * k) / Math.max(1, ma.size);
            }
            const dx = withCov(P1, K).X;
            const L = layers.find(l => l.id === id); L.bwdOut = varAll(G); L.bwdIn = varAll(dx);
            addGrad(src[0], dx); break;
          }
          const c = t.conv, RI = c.RI, RO = c.RO, P2 = RO.gh * RO.gw, P1 = RI.gh * RI.gw, k = c.k, s0 = c.s, pad = c.pad;
          const Gm = moments({ T: P2, X: G });
          let K;
          if (!RI.torus) {
            // pixel by pixel: input pixels a, b receive output gradients through the same kernel offset
            K = new Float64Array(P1 * P1);
            for (let P = 0; P < P2; P++) for (let Q = 0; Q < P2; Q++) {
              const g = Gm[P * P2 + Q]; if (!g) continue;
              const pi = Math.floor(P / RO.gw), pj = P % RO.gw, qi = Math.floor(Q / RO.gw), qj = Q % RO.gw;
              for (let ti = 0; ti < k; ti++) { if (!valid(pi, ti, s0, pad, RI.h) || !valid(qi, ti, s0, pad, RI.h)) continue; for (let tj = 0; tj < k; tj++) { if (!valid(pj, tj, s0, pad, RI.w) || !valid(qj, tj, s0, pad, RI.w)) continue; K[((pi * s0 - pad + ti) * RI.gw + (pj * s0 - pad + tj)) * P1 + (qi * s0 - pad + ti) * RI.gw + (qj * s0 - pad + tj)] += g; } }
            }
            for (let i = 0; i < K.length; i++) K[i] *= c.f;
          } else if (!RO.torus) {
            // a wrapped input under an exact output: the covariance by actual displacement, averaged over the map
            const Rh = Math.floor(near(RI.gh) / RI.dh), Rw = Math.floor(near(RI.gw) / RI.dw), W2 = 2 * Rw + 1, B = new Float64Array((2 * Rh + 1) * W2); let farSum = 0;
            for (let P = 0; P < P2; P++) for (let Q = 0; Q < P2; Q++) {
              const pi = Math.floor(P / RO.gw), pj = P % RO.gw, qi = Math.floor(Q / RO.gw), qj = Q % RO.gw; let nh = 0, nw = 0;
              for (let t2 = 0; t2 < k; t2++) { if (valid(pi, t2, s0, pad, RI.h) && valid(qi, t2, s0, pad, RI.h)) nh++; if (valid(pj, t2, s0, pad, RI.w) && valid(qj, t2, s0, pad, RI.w)) nw++; }
              const v = nh * nw * Gm[P * P2 + Q], Dh = (qi - pi) * s0, Dw = (qj - pj) * s0;
              if (Math.abs(Dh) <= Rh && Math.abs(Dw) <= Rw) B[(Dh + Rh) * W2 + Dw + Rw] += v; else farSum += v;
            }
            const pairs = (n, R) => { let c2 = 0; for (let d = -R; d <= R; d++) c2 += Math.max(0, n - Math.abs(d)); return c2; };
            const farPairs = (RI.h * RI.w) ** 2 - pairs(RI.h, Rh) * pairs(RI.w, Rw);
            K = statCov(RI, (dh, dw) => { dh = Math.round(dh); dw = Math.round(dw); return Math.abs(dh) <= Rh && Math.abs(dw) <= Rw ? c.f * B[(dh + Rh) * W2 + dw + Rw] / Math.max(1, (RI.h - Math.abs(dh)) * (RI.w - Math.abs(dw))) : farPairs > 0 ? c.f * farSum / farPairs : 0; });
          } else {
            // wrapped in and out: pixels the same number of strides apart share kernel offsets; others don't
            const gst = stat({ img: RO }, G), cbar = readShare(RI.h, RO.h, k, s0, pad) * readShare(RI.w, RO.w, k, s0, pad);
            const Rh = Math.floor(near(RI.gh) / RI.dh), Rw = Math.floor(near(RI.gw) / RI.dw);
            K = statCov(RI, (dh, dw) => { dh = Math.round(dh); dw = Math.round(dw); if (Math.abs(dh) > Rh || Math.abs(dw) > Rw) return c.f * cbar * gst.far / (s0 * s0); return dh % s0 || dw % s0 ? 0 : c.f * cbar * kat(gst, dh / s0, dw / s0); });
          }
          const dx = withCov(P1, K).X;
          const L = layers.find(l => l.id === id); L.bwdOut = varAll(G); L.bwdIn = varAll(dx);
          addGrad(src[0], dx);
          break;
        }
        case 'pool': {
          const dx = new Float32Array(A.X.length);
          if (t.gap) { for (let q = 0; q < A.T; q++) for (let s2 = 0; s2 < NS; s2++) dx[q * NS + s2] = G[s2] / t.gap; }
          else if (t.win) t.win.forEach((cells, q) => { for (const c of cells) for (let s2 = 0; s2 < NS; s2++) dx[c * NS + s2] += G[q * NS + s2] / cells.length; });
          else {
            const pl = t.pool;
            for (let q = 0; q < pl.wins.length; q++) for (let s2 = 0; s2 < NS; s2++) {
              const g = G[q * NS + s2] * pl.scale;
              if (pl.mx) { const a = pl.arg[q * NS + s2]; if (a >= 0) dx[a * NS + s2] += g; }
              else pl.wins[q].forEach((a, j) => { if (!pl.use || pl.use[(q * NS + s2) * pl.k * pl.k + j]) dx[a * NS + s2] += g / (pl.k * pl.k); });
            }
          }
          addGrad(src[0], dx);
          break;
        }
        case 'upsample': {
          const u = t.up, RI = u.RI, RO = u.RO, f = u.f, P1 = RI.gh * RI.gw, dx = new Float32Array(P1 * NS);
          if (!u.src) {
            // each input pixel collects the f × f output pixels over it
            for (let pi = 0; pi < RI.gh; pi++) for (let pj = 0; pj < RI.gw; pj++) for (let ri = 0; ri < f; ri++) for (let rj = 0; rj < f; rj++) {
              const q = ((pi + ri * RO.dh) % RO.gh) * RO.gw + ((pj + rj * RO.dw) % RO.gw), p0 = pi * RI.gw + pj;
              for (let s2 = 0; s2 < NS; s2++) dx[p0 * NS + s2] += G[q * NS + s2];
            }
          } else {
            const cov = new Uint8Array(P1); let hc = 1, wc = 1;
            for (let q = 0; q < u.src.length; q++) { const a = u.src[q]; cov[a] = 1; hc = Math.max(hc, Math.floor(a / RI.gw) + 1); wc = Math.max(wc, (a % RI.gw) + 1); for (let s2 = 0; s2 < NS; s2++) dx[a * NS + s2] += G[q * NS + s2]; }
            // pixels outside the simulated patch: an independent copy of a covered pixel's gradient
            for (let a = 0; a < P1; a++) if (!cov[a]) { const b = (Math.floor(a / RI.gw) % hc) * RI.gw + ((a % RI.gw) % wc), perm = permuted(); for (let s2 = 0; s2 < NS; s2++) dx[a * NS + s2] = dx[b * NS + perm[s2]]; }
          }
          addGrad(src[0], dx);
          break;
        }
        case 'act': case 'func': {
          if (t && t.pass) { addGrad(src[0], G); break; }
          if (t && t.none) break;
          if (t && t.catPos) { let o = 0; t.catPos.forEach((T2, j) => { addGrad(src[j], G.slice(o * NS, (o + T2) * NS)); o += T2; }); break; }
          if (t && t.pick) { const dX = new Float32Array(t.T0 * NS); t.pick.forEach((u, k) => { for (let s2 = 0; s2 < NS; s2++) dX[u * NS + s2] += G[k * NS + s2]; }); addGrad(src[0], dX); break; }
          if (t && t.ksum) { const K = t.ksum, dX = new Float32Array(G.length * K); for (let i = 0; i < G.length; i++) for (let k = 0; k < K; k++) dX[i * K + k] = G[i]; addGrad(src[0], dX); break; }
          if (t && t.seqOf) {
            // each grid pixel's gradient: for each sample, the gradient of one of its copies, chosen at random (the
            // copies stand for different pixels, so this keeps the gradient's spread over the map)
            const { src: sp, P1 } = t.seqOf, byQ = new Map(), dx = new Float32Array(P1 * NS);
            sp.forEach(([q, pm], a) => { if (!byQ.has(q)) byQ.set(q, []); byQ.get(q).push([a, pm]); });
            for (const [q, list] of byQ) {
              const inv = list.map(([, pm]) => { if (!pm) return null; const v = new Int32Array(NS); for (let s2 = 0; s2 < NS; s2++) v[pm[s2]] = s2; return v; });
              for (let s2 = 0; s2 < NS; s2++) { const c = list.length > 1 ? Math.floor(rand() * list.length) : 0; dx[q * NS + s2] = G[list[c][0] * NS + (inv[c] ? inv[c][s2] : s2)]; }
            }
            addGrad(src[0], dx); break;
          }
          if (t && t.flat) { const P = t.flat, dx = new Float32Array(P * NS); if (P === 1) dx.set(G); else for (let q = 0; q < P; q++) { const perm = permuted(); for (let s2 = 0; s2 < NS; s2++) dx[q * NS + s2] = G[perm[s2]]; } addGrad(src[0], dx); break; }
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
          if (t && t.kb) {
            const { K, STR, A: Av, B: Bv } = t.kb, T = me.T, so = shapeOf(id) || [], N = so[so.length - 1] || K, D2 = so[so.length - 2] || 1;
            const get = v => (v.K > 1 ? (t2, s2, k) => v.X[((v.T === 1 ? 0 : t2) * NS + s2) * K + k] : v.bc === 'inner' ? (t2, s2, k) => v.X[(v.T === 1 ? 0 : t2) * NS + ((s2 + k * STR) % NS)] : (t2, s2) => v.X[(v.T === 1 ? 0 : t2) * NS + s2]);
            const ga = get(Av), gb = get(Bv);
            const part = [(g, a, b) => g * ({ '+': 1, '-': 1, '*': b, '/': 1 / b }[op]), (g, a, b) => ({ '+': g, '-': -g, '*': g * a, '/': -g * a / (b * b) }[op])];
            [Av, Bv].forEach((v, j) => {
              if (v.param || !src[j]) return;
              const Y = new Float32Array((v.T === 1 ? 1 : T) * NS * (v.K > 1 ? K : 1));
              for (let t2 = 0; t2 < T; t2++) for (let s2 = 0; s2 < NS; s2++) for (let k = 0; k < K; k++) {
                const i = (t2 * NS + s2) * K + k, g = part[j](G[i], ga(t2, s2, k), gb(t2, s2, k)), u = v.T === 1 ? 0 : t2;
                if (v.K > 1) Y[(u * NS + s2) * K + k] += g;
                else if (v.bc === 'inner') Y[u * NS + ((s2 + k * STR) % NS)] += g;
                else Y[u * NS + s2] += g;
              }
              // the K entries stand for N (a unit's value used along the whole axis), or for every unit (B, C: used by
              // all D2 of them): scaled up to that many independent terms
              const sc = v.K > 1 ? 1 : Math.sqrt((v.bc === 'inner' ? D2 : N) / K); if (sc !== 1) for (let i = 0; i < Y.length; i++) Y[i] *= sc;
              addGrad(src[j], Y);
            });
            break;
          }
          const B0 = ins[1], AL = t && t.aligned;
          const B = AL ? AL[1] : B0;
          if (!B0) {
            const c = t.c, X = A.X;
            const d = { '+': () => 1, '-': () => (p.rev ? -1 : 1), '*': () => c, '/': i => (p.rev ? -c / (X[i] * X[i]) : 1 / c) }[op];
            addGrad(src[0], mapG(G, (g, i) => g * d(i)));
            break;
          }
          const A1 = AL ? AL[0] : A, T = me.k === 's' ? 1 : me.T, at = i => (A1.k === 's' || A1.T === T ? i : i % NS), bt = i => (B.k === 's' || B.T === T ? i : i % NS);
          // an input broadcast along a feature dimension of size K (x.unsqueeze(-1) ⊙ a K-wide tensor) is used K
          // times: its gradient is the sum of K terms, independent ones (other units), simulated as for a sum
          const fan = s0 => {
            const so = shapeOf(id), si = shapeOf(s0); if (me.k !== 'f' || !so || !si) return 1;
            let K = 1; const first = so.length >= 3 ? 2 : 1;
            for (let j = first; j < so.length; j++) { const d = si[si.length - (so.length - j)]; if (so[j] > 1 && (d === 1 || (d === undefined && si.length >= 2))) K *= so[j]; }
            return K;
          };
          const put = (s0, Gx) => {
            const K = fan(s0); if (K <= 1) { addGrad(s0, Gx); return; }
            const Kc = Math.min(K, 16), sc = Math.sqrt(K / Kc), Y = new Float32Array(Gx.length);
            for (let c = 0; c < Kc; c++) { const pm = c ? permuted() : null; for (let tt = 0; tt * NS < Gx.length; tt++) for (let s2 = 0; s2 < NS; s2++) Y[tt * NS + s2] += sc * Gx[tt * NS + (pm ? pm[s2] : s2)]; }
            addGrad(s0, Y);
          };
          if (op === '+') { put(src[0], G); put(src[1], G); }
          else if (op === '-') { put(src[0], G); put(src[1], mapG(G, g => -g)); }
          else if (op === '*') { put(src[0], mapG(G, (g, i) => g * B.X[bt(i)])); put(src[1], mapG(G, (g, i) => g * A1.X[at(i)])); }
          else if (op === '/') { put(src[0], mapG(G, (g, i) => g / B.X[bt(i)])); put(src[1], mapG(G, (g, i) => (-g * A1.X[at(i)]) / (B.X[bt(i)] * B.X[bt(i)]))); }
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
