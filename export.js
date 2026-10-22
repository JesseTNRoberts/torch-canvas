// ───────────────────────── Torch Canvas: export model.py + train.py ─────────────────────────
// model.py is the code panel's code (forward, loss, predict). train.py is a complete training script
// written from the export dialog's selections: the values are constants at the top (BATCH_SIZE, LR, ...)
// and only the chosen optimizer, schedule, precision and options appear in the code.
// Runs in the browser and in Node.
const NNExport = (() => {
  'use strict';
  const C = typeof NN !== 'undefined' ? NN : require('./core.js');

  const OPTIMIZERS = { sgd: 'SGD', adam: 'Adam', adamw: 'AdamW', rmsprop: 'RMSprop' };
  const SCHEDULES = { constant: 'Constant', cosine: 'Cosine decay', linear: 'Linear decay', step: 'Step decay', onecycle: 'One-cycle' };
  const PRECISIONS = { fp32: 'fp32', bf16: 'bf16 mixed precision', fp16: 'fp16 mixed precision' };

  // sensible starting points: the transformer recipe when the model has attention, plain Adam otherwise
  function defaults(g) {
    const attn = (g.nodes || []).some(n => ['heads', 'mask', 'rope'].includes(n.type)) || Object.values(g.blocks || {}).some(b => b.graph.nodes.some(n => ['heads', 'mask', 'rope'].includes(n.type)));
    const t = g.symbols && g.symbols.t != null ? +g.symbols.t : 64;
    return {
      optimizer: attn ? 'adamw' : 'adam', lr: attn ? 3e-4 : 1e-3, weight_decay: attn ? 0.1 : 0, beta1: 0.9, beta2: attn ? 0.95 : 0.999, eps: 1e-8,
      momentum: 0.9, nesterov: false, rms_alpha: 0.99, no_decay: true,
      schedule: attn ? 'cosine' : 'constant', warmup: attn ? 100 : 0, min_lr_ratio: 0.1, step_size: 1000, gamma: 0.1,
      batch_size: 32, seq_len: t, num_samples: 2048, val_fraction: 0.1,
      length: 'epochs', epochs: 10, max_steps: 1000, grad_accum: 1, grad_clip: attn ? 1.0 : 0, precision: 'fp32', device: 'auto', seed: 0,
      log_every: 50, eval_every: 200, save_every: 1000, out_dir: 'checkpoints',
      wandb: false, wandb_project: String(g.name || 'model').replace(/[^\w-]/g, '_'),
      compile: false, grad_ckpt: false, ema: 0, resume: false,
    };
  }

  const py = v => (typeof v === 'boolean' ? (v ? 'True' : 'False') : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : String(v)) : JSON.stringify(String(v)));
  const num = (v, d) => (v === '' || v == null || !isFinite(+v) ? d : +v);
  const ident = s => { let t = String(s || '').replace(/[^\w]/g, '_').replace(/^(\d)/, '_$1') || 'x'; return t; };
  const clsName = s => { const t = ident(s || 'Model'); return t.charAt(0).toUpperCase() + t.slice(1); };

  // What the data looks like and how one batch becomes a loss: read from the canvas.
  function plan(g) {
    const errors = [];
    const lp = C.lossPlan(g);
    if (!lp) return { errors: ['Add a loss from Components → Loss, so train.py knows what to minimize.'] };
    errors.push(...lp.warnings);
    if (!lp.terms.length) errors.push('None of the losses is connected to a model output and a target.');
    const { conflicts, errors: errs } = C.inferShapes(g);
    if (conflicts.size) errors.push('Fix the size conflicts first (marked in red).');
    if (errs.size) errors.push([...errs.values()][0]);
    if (errors.length) return { errors };
    const { ix, shapes, sym } = lp;
    // position sizes: input dims other than the batch and the features (a token input's dims after the batch)
    const posSyms = new Set();
    const tokOf = n => (ix.outs.get(n.id) || []).map(o => ix.byId.get(o.to)).find(c => c && c.type === 'embed' && !c.params.positions);
    for (const n of g.nodes) if (n.type === 'input') { const d = C.parseShape(n.params.shape); (tokOf(n) ? d.slice(1) : d.slice(1, -1)).forEach(x => { if (C.identsIn(x).length) posSyms.add(String(x)); }); }
    const perSample = sh => sh.slice(1).map(d => (posSyms.has(String(d)) ? 'SEQ_LEN' : String(C.evalDim(d, sym) ?? d)));
    const tup = a => `(${a.join(', ')}${a.length === 1 ? ',' : ''})`;
    // dataset tensors: the forward inputs, then targets that aren't forward inputs (masked labels are made per batch)
    const data = [];
    for (const n of lp.fwdInputs) {
      const name = ident(n.params.name), dims = perSample(C.parseShape(n.params.shape)), tok = tokOf(n);
      const vocab = tok ? C.evalDim(tok.params.num, sym) : null;
      data.push({ name, expr: tok ? `torch.randint(0, ${vocab}, ${tup(['n', ...dims])})` : `torch.randn(${['n', ...dims].join(', ')})`, what: tok ? `token ids in [0, ${vocab})` : 'floats', vocab });
    }
    const masked = lp.terms.find(t => t.mode === 'masked');
    for (const a of lp.args.slice(lp.outputs.length)) {
      if (data.some(d => d.name === a) || a === 'labels' && masked) continue;
      const t = lp.terms.find(z => z.targetName === a); if (!t) continue;
      const P = shapes.get(ix.ins.get(t.outputId)[0]) || [], pd = perSample(P);
      const C_ = pd[pd.length - 1];
      const expr = t.fn === 'cross_entropy' ? `torch.randint(0, ${C_}, ${tup(['n', ...pd.slice(0, -1)])})` : t.fn === 'bce' ? `torch.randint(0, 2, ${tup(['n', ...pd])}).float()` : `torch.randn(${['n', ...pd].join(', ')})`;
      data.push({ name: a, expr, what: t.fn === 'cross_entropy' ? `class indices in [0, ${C_})` : t.fn === 'bce' ? '0/1 labels' : 'targets', target: true });
    }
    let mask = null;
    if (masked) {
      const src = g.nodes.find(n => n.id === masked.target), tok = src && tokOf(src);
      const vocab = tok ? C.evalDim(tok.params.num, sym) : null;
      mask = { input: ident(src ? src.params.name : 'input_ids'), vocab, prob: masked.maskProb, id: masked.maskId !== '' && masked.maskId != null ? +masked.maskId : (vocab ? vocab - 1 : 0) };
    }
    const blockClasses = [...new Set((C.expandBlocks(g).flat.groups || []).map(gr => clsName(gr.cls)))];
    return { errors: [], lp, data, mask, hasSeq: posSyms.size > 0, fwd: lp.fwdInputs.map(n => ident(n.params.name)), outs: lp.outputs.map(o => ident(o.params.name || 'out')), args: lp.args, blockClasses };
  }

  // Python number literals: 3e-4 rather than 0.0003, and floats keep a decimal point
  const pf = v => { v = +v; if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential().replace('e-', 'e-'); return Number.isInteger(v) ? v.toFixed(1) : String(v); };
  const pi = v => String(Math.round(+v));

  // train.py is written from the selections: only the chosen optimizer, schedule and options appear,
  // and the values sit in a block of constants at the top.
  function trainScript(g, s0, opts = {}) {
    const P = plan(g);
    if (P.errors.length) return { errors: P.errors };
    const s = Object.assign(defaults(g), s0 || {});
    const name = clsName(opts.className || g.name), mod = opts.moduleName || 'model';
    const Q = '"' + '"' + '"';
    const L = [], push = (...a) => L.push(...a);

    // ── what was chosen ──
    const opt = OPTIMIZERS[s.optimizer] ? s.optimizer : 'adamw';
    const sched = SCHEDULES[s.schedule] ? s.schedule : 'constant';
    const prec = PRECISIONS[s.precision] ? s.precision : 'fp32';
    const lr = num(s.lr, 3e-4), wd = Math.max(0, num(s.weight_decay, 0)), warm = Math.max(0, Math.round(num(s.warmup, 0)));
    const accum = Math.max(1, Math.round(num(s.grad_accum, 1))), clip = Math.max(0, num(s.grad_clip, 0)), emaDecay = num(s.ema, 0);
    const ema = emaDecay > 0 && emaDecay < 1;
    const valFrac = Math.min(0.9, Math.max(0, num(s.val_fraction, 0.1)));
    const val = valFrac > 0, evalEvery = Math.max(0, Math.round(num(s.eval_every, 0))), saveEvery = Math.max(0, Math.round(num(s.save_every, 0)));
    const steps = s.length === 'steps';
    const groups = wd > 0 && s.no_decay !== false;
    const ckpt = !!s.grad_ckpt && P.blockClasses.length > 0, compile = !!s.compile;
    const amp = prec !== 'fp32', fp16 = prec === 'fp16';
    const hasSched = sched !== 'constant' || warm > 0;
    const usesTotal = ['cosine', 'linear', 'onecycle'].includes(sched);
    const fwd = compile ? 'compiled' : 'model';
    const wandb = !!s.wandb, resume = !!s.resume, useWhile = steps || resume;
    const momentum = num(s.momentum, opt === 'sgd' ? 0.9 : 0), eps = num(s.eps, 1e-8);
    const b1 = num(s.beta1, 0.9), b2 = num(s.beta2, 0.999);

    // ── header and imports ──
    push(`${Q}Train ${name}: written by Torch Canvas, with model.py next to it.`, '',
      'The settings below come from the export dialog; change them right here. As written, the script trains',
      'on stand-in random data with the right shapes, so you can check everything runs, then swap', `make_dataset() for your own data.${Q}`);
    push(...[ckpt && 'import functools', 'import os', 'import time'].filter(Boolean), '', 'import torch');
    if (groups) push('import torch.nn as nn');
    push(`from torch.utils.data import DataLoader, TensorDataset${val ? ', random_split' : ''}`);
    if (ema) push('from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn');
    if (ckpt) push('from torch.utils.checkpoint import checkpoint');
    if (wandb) push('import wandb  # pip install wandb');
    push('', `from ${mod} import ${[name, ...(ckpt ? P.blockClasses.filter(c => c !== name) : [])].join(', ')}`, '', '');

    // ── settings ──
    const set = [], c = (k, v, note) => set.push([k, v, note]);
    c('BATCH_SIZE', pi(num(s.batch_size, 32)), accum > 1 ? 'examples per micro-batch' : 'examples per step');
    if (P.hasSeq) c('SEQ_LEN', pi(num(s.seq_len, 64)), 'tokens per example');
    c('NUM_SAMPLES', pi(num(s.num_samples, 2048)), 'size of the stand-in dataset');
    if (val) c('VAL_FRACTION', pf(valFrac), 'share held out for validation');
    if (P.mask) { c('MASK_PROB', pf(P.mask.prob), 'share of tokens masked'); c('MASK_ID', pi(P.mask.id), 'the [MASK] token id'); }
    set.push(null);
    c('LR', pf(lr), sched === 'constant' ? '' : 'peak learning rate');
    if (wd > 0) c('WEIGHT_DECAY', pf(wd));
    if (warm > 0) c('WARMUP_STEPS', pi(warm));
    if (sched === 'cosine' || sched === 'linear') c('MIN_LR_RATIO', pf(num(s.min_lr_ratio, 0.1)), 'the learning rate decays to this share of LR');
    if (sched === 'step') { c('STEP_SIZE', pi(num(s.step_size, 1000)), 'steps between drops'); c('GAMMA', pf(num(s.gamma, 0.1)), 'factor per drop'); }
    if (steps) c('MAX_STEPS', pi(num(s.max_steps, 1000)), 'optimizer steps'); else c('EPOCHS', pi(num(s.epochs, 10)));
    if (accum > 1) c('GRAD_ACCUM', pi(accum), 'micro-batches per optimizer step');
    if (clip > 0) c('GRAD_CLIP', pf(clip), 'max gradient norm');
    if (ema) c('EMA_DECAY', pf(emaDecay));
    c('SEED', pi(num(s.seed, 0)));
    set.push(null);
    c('LOG_EVERY', pi(Math.max(1, num(s.log_every, 50))), 'steps');
    if (val && evalEvery) c('EVAL_EVERY', pi(evalEvery), 'steps');
    if (saveEvery) c('SAVE_EVERY', pi(saveEvery), 'steps');
    c('OUT_DIR', py(s.out_dir || 'checkpoints'));
    if (resume) c('RESUME', 'None', 'a checkpoint to continue from, e.g. "checkpoints/last.pt"');
    if (wandb) c('WANDB_PROJECT', py(s.wandb_project || name));
    push('# ── settings (from the export dialog) ──');
    const w = Math.max(...set.filter(Boolean).map(([k, v]) => k.length + v.length + 3));
    for (const e of set) {
      if (!e) { push(''); continue; }
      const [k, v, note] = e, line = `${k} = ${v}`;
      push(note ? line + ' '.repeat(w - line.length + 2) + '# ' + note : line);
    }
    push('', '');

    // ── data ──
    const tensors = P.data.map(d => d.name);
    push('def make_dataset():', `    ${Q}Stand-in random data with the shapes ${name} expects. Replace it with your own Dataset whose`,
      `    items are (${tensors.join(', ')}):`);
    for (const d of P.data) push(`        ${d.name}: ${d.what}${d.target ? ' (target)' : ''}`);
    if (P.mask) push(`    Masked-token labels are made from ${P.mask.input} batch by batch (mask_tokens), so they aren't stored.`);
    else if (P.lp.terms.some(t => t.mode === 'next')) push(`    Next-token targets are the inputs themselves, shifted by one inside loss().`);
    push(`    ${Q}`, '    n = NUM_SAMPLES');
    for (const d of P.data) push(`    ${d.name} = ${d.expr}`);
    push(`    return TensorDataset(${tensors.join(', ')})`, '', '');
    if (P.mask) {
      push('def mask_tokens(ids):', `    ${Q}BERT-style masking: picks MASK_PROB of the tokens; of those, 80% become MASK_ID, 10% a random`,
        `    token and 10% stay as they are. Labels hold the original token at picked positions and -100 elsewhere.${Q}`,
        '    labels = ids.clone()', '    picked = torch.rand(ids.shape, device=ids.device) < MASK_PROB', '    labels[~picked] = -100',
        '    ids = ids.clone()', '    r = torch.rand(ids.shape, device=ids.device)', '    ids[picked & (r < 0.8)] = MASK_ID',
        '    swap = picked & (r >= 0.8) & (r < 0.9)', `    ids[swap] = torch.randint(0, ${P.mask.vocab}, (int(swap.sum()),), device=ids.device)`, '    return ids, labels', '', '');
    }
    push('def compute_loss(model, batch):', `    ${Q}One batch: forward(), then loss() with the targets.${Q}`,
      `    ${tensors.join(', ')}${tensors.length === 1 ? ',' : ''} = batch`);
    if (P.mask) push(`    ${P.mask.input}, labels = mask_tokens(${P.mask.input})`);
    push(`    ${P.outs.join(', ')} = model(${P.fwd.join(', ')})`, `    return model.loss(${P.args.join(', ')})`, '', '');

    // ── precision ──
    if (amp) {
      const doc = { bf16: 'bf16 mixed precision: matrix multiplies in bf16, weights and sensitive ops in fp32.', fp16: 'fp16 mixed precision (with a GradScaler in the loop).' }[prec];
      const body = `torch.autocast(device.type, dtype=torch.${prec === 'bf16' ? 'bfloat16' : 'float16'})`;
      push('def autocast(device):', `    ${Q}${doc}${Q}`, `    return ${body}`, '', '');
    }

    // ── optimizer ──
    const kw = ['lr=LR'];
    if (opt === 'adam' || opt === 'adamw') { if (b1 !== 0.9 || b2 !== 0.999) kw.push(`betas=(${pf(b1).replace(/\.0$/, '.0')}, ${pf(b2)})`); if (eps !== 1e-8) kw.push(`eps=${pf(eps)}`); }
    if (opt === 'sgd' && momentum > 0) { kw.push(`momentum=${pf(momentum)}`); if (s.nesterov) kw.push('nesterov=True'); }
    if (opt === 'rmsprop') { if (num(s.rms_alpha, 0.99) !== 0.99) kw.push(`alpha=${pf(num(s.rms_alpha, 0.99))}`); if (eps !== 1e-8) kw.push(`eps=${pf(eps)}`); if (momentum > 0) kw.push(`momentum=${pf(momentum)}`); }
    const ctor = `torch.optim.${OPTIMIZERS[opt]}`;
    let optLine;
    if (groups) {
      push('def make_optimizer(model):', `    ${Q}${OPTIMIZERS[opt]} with weight decay on the weight matrices only: biases, norm weights and embeddings`, `    are left undecayed.${Q}`,
        '    embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}',
        '    decay = [p for p in model.parameters() if p.dim() >= 2 and id(p) not in embeddings]',
        '    no_decay = [p for p in model.parameters() if p.dim() < 2 or id(p) in embeddings]',
        '    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}]',
        `    return ${ctor}(groups, ${kw.join(', ')})`, '', '');
      optLine = 'optimizer = make_optimizer(model)';
    } else {
      if (wd > 0) kw.push('weight_decay=WEIGHT_DECAY'); else if (opt === 'adamw') kw.push('weight_decay=0.0');
      optLine = `optimizer = ${ctor}(model.parameters(), ${kw.join(', ')})`;
    }

    // ── schedule (PyTorch's own schedulers) ──
    const LRS = 'torch.optim.lr_scheduler';
    let schedLines = [];
    if (hasSched) {
      const rest = warm > 0 ? 'max(1, total_steps - WARMUP_STEPS)' : 'total_steps';
      const warmup = `${LRS}.LinearLR(optimizer, start_factor=1 / WARMUP_STEPS, total_iters=WARMUP_STEPS)`;
      let decay = null, note = '';
      if (sched === 'cosine') { decay = `${LRS}.CosineAnnealingLR(optimizer, T_max=${rest}, eta_min=LR * MIN_LR_RATIO)`; note = 'cosine decay to MIN_LR_RATIO * LR'; }
      if (sched === 'linear') { decay = `${LRS}.LinearLR(optimizer, start_factor=1.0, end_factor=MIN_LR_RATIO, total_iters=${rest})`; note = 'linear decay to MIN_LR_RATIO * LR'; }
      if (sched === 'step') { decay = `${LRS}.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)`; note = 'multiply by GAMMA every STEP_SIZE steps'; }
      if (sched === 'onecycle') {
        const pct = warm > 0 ? 'min(0.5, max(2 / total_steps, WARMUP_STEPS / total_steps))' : '0.3';
        schedLines = [`# one-cycle: warm up to LR over the first ${warm > 0 ? 'WARMUP_STEPS' : '30% of the'} steps, then anneal`,
          `scheduler = ${LRS}.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=${pct})`];
      } else if (decay && warm > 0) {
        schedLines = [`# linear warmup over WARMUP_STEPS, then ${note}`, `warmup = ${warmup}`, `decay = ${decay}`,
          `scheduler = ${LRS}.SequentialLR(optimizer, [warmup, decay], milestones=[WARMUP_STEPS])`];
      } else if (decay) schedLines = [`# ${note}`, `scheduler = ${decay}`];
      else schedLines = ['# linear warmup over WARMUP_STEPS, then constant', `scheduler = ${warmup}`];
    }

    // ── evaluation ──
    const autoWith = (ind, line) => (amp ? [`${ind}with autocast(device):`, `${ind}    ${line}`] : [`${ind}${line}`]);
    if (val) {
      push('@torch.no_grad()', 'def evaluate(model, loader, device):', '    model.eval()', '    total = 0.0', '    for batch in loader:',
        '        batch = [b.to(device) for b in batch]', ...autoWith('        ', 'total += compute_loss(model, batch).item()'),
        '    model.train()', '    return total / len(loader)', '', '');
    }
    if (ckpt) {
      const cls = P.blockClasses.length === 1 ? P.blockClasses[0] : `(${P.blockClasses.join(', ')})`;
      push('def checkpoint_blocks(model):', `    ${Q}Gradient checkpointing: recompute each block's activations in the backward pass instead of`,
        `    keeping them. Much less memory for roughly a third more compute.${Q}`,
        '    for m in model.modules():', `        if isinstance(m, ${cls}):`,
        '            m.forward = functools.partial(checkpoint, m.forward, use_reentrant=False)', '', '');
    }

    // ── main ──
    const M = [];
    const m = (...a) => M.push(...a.map(x => (x === '' ? '' : '    ' + x)));
    m('torch.manual_seed(SEED)');
    if (s.device && s.device !== 'auto') m(`device = torch.device(${py(s.device)})`);
    else m('device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")');
    m('', `model = ${name}().to(device)`);
    if (ckpt) m('checkpoint_blocks(model)');
    m('print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}")');
    if (compile) m('compiled = torch.compile(model)  # used for the forward passes; model keeps the weights');
    m('', 'data = make_dataset()');
    if (val) {
      m('n_val = max(1, int(len(data) * VAL_FRACTION))',
        'train_set, val_set = random_split(data, [len(data) - n_val, n_val], generator=torch.Generator().manual_seed(SEED))',
        'train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)',
        'val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)');
    } else m('train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)');
    if (steps) m(`total_steps = MAX_STEPS`);
    else m(`total_steps = EPOCHS * ${accum > 1 ? '(len(train_loader) // GRAD_ACCUM)' : 'len(train_loader)'}`);
    m('', optLine, ...schedLines);
    if (fp16) m('scaler = torch.amp.GradScaler(device.type)  # keeps small fp16 gradients from underflowing');
    if (ema) m('ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))  # moving average of the weights');
    m('', 'step = 0');
    if (resume) {
      m('if RESUME:', '    ckpt = torch.load(RESUME, map_location=device)', '    model.load_state_dict(ckpt["model"])', '    optimizer.load_state_dict(ckpt["optimizer"])');
      if (hasSched) m('    scheduler.load_state_dict(ckpt["scheduler"])');
      if (ema) m('    ema.load_state_dict(ckpt["ema"])');
      m('    step = ckpt["step"]', '    print(f"resumed from {RESUME} at step {step}")');
    }
    if (wandb) {
      const cfg = set.filter(e => e && !['OUT_DIR', 'RESUME', 'WANDB_PROJECT'].includes(e[0])).map(e => e[0]);
      m('', 'wandb.init(project=WANDB_PROJECT, config={');
      for (let i = 0; i < cfg.length; i += 4) m('    ' + cfg.slice(i, i + 4).map(k => `"${k.toLowerCase()}": ${k},`).join(' '));
      m(`    "optimizer": ${py(OPTIMIZERS[opt])}, "schedule": ${py(sched)}, "precision": ${py(prec)},`, '})');
    }
    m('', 'def log(metrics):', '    print(f"step {step:>6}  " + "  ".join(f"{k} {v:.4g}" for k, v in metrics.items()))');
    if (wandb) m('    wandb.log(metrics, step=step)');
    m('', 'def save():', '    os.makedirs(OUT_DIR, exist_ok=True)', '    path = os.path.join(OUT_DIR, "last.pt")',
      `    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), ${hasSched ? '"scheduler": scheduler.state_dict(), ' : ''}${ema ? '"ema": ema.state_dict(), ' : ''}"step": step}, path)`,
      '    return path', '');
    if (val) {
      m('def validate():', `    metrics = {"val/loss": evaluate(${fwd}, val_loader, device)}`);
      if (ema) m('    metrics["val/loss_ema"] = evaluate(ema.module, val_loader, device)');
      m('    log(metrics)', '');
    }
    m('model.train()', 'running, count, t0 = 0.0, 0, time.time()');
    // the loop: a plain epoch loop, or a step-counted one for a fixed number of steps (or a resumed run)
    m(useWhile ? 'while step < total_steps:' : 'for epoch in range(EPOCHS):');
    const I = '            ';
    const loop = [];
    const l = (...a) => loop.push(...a.map(x => (x === '' ? '' : I + x)));
    if (accum > 1) {
      loop.push('        for i, batch in enumerate(train_loader):');
      l('batch = [b.to(device) for b in batch]');
      loop.push(...autoWith(I, `loss = compute_loss(${fwd}, batch) / GRAD_ACCUM`));
      l(fp16 ? 'scaler.scale(loss).backward()' : 'loss.backward()', 'running += loss.item()', 'if (i + 1) % GRAD_ACCUM:', '    continue  # keep accumulating gradients');
    } else {
      loop.push('        for batch in train_loader:');
      l('batch = [b.to(device) for b in batch]');
      loop.push(...autoWith(I, `loss = compute_loss(${fwd}, batch)`));
      l(fp16 ? 'scaler.scale(loss).backward()' : 'loss.backward()', 'running += loss.item()');
    }
    if (clip > 0) { if (fp16) l('scaler.unscale_(optimizer)'); l('grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)'); }
    if (fp16) l('scaler.step(optimizer)', 'scaler.update()'); else l('optimizer.step()');
    l('optimizer.zero_grad(set_to_none=True)');
    if (hasSched) l('scheduler.step()');
    if (ema) l('ema.update_parameters(model)');
    l('step += 1', 'count += 1', '', 'if step % LOG_EVERY == 0:');
    l(`    log({"train/loss": running / count, "lr": optimizer.param_groups[0]["lr"], ${clip > 0 ? '"grad_norm": grad_norm.item(), ' : ''}"sec/step": (time.time() - t0) / count})`,
      '    running, count, t0 = 0.0, 0, time.time()');
    // periodic validation and checkpoints; the last step's are done once, after the loop
    if (val && evalEvery) l('if step % EVAL_EVERY == 0 and step < total_steps:', '    validate()');
    if (saveEvery) l('if step % SAVE_EVERY == 0 and step < total_steps:', '    save()');
    if (useWhile) l('if step == total_steps:', '    break');
    M.push(...loop);
    m('');
    if (val) m('validate()');
    m('print("saved", save())');
    if (wandb) m('wandb.finish()');
    push('def main():', ...M, '', '', 'if __name__ == "__main__":', '    main()');
    return { errors: [], train: L.join('\n') + '\n', settings: s, plan: P };
  }

  return { defaults, trainScript, plan, OPTIMIZERS, SCHEDULES, PRECISIONS };
})();
if (typeof module !== 'undefined') module.exports = NNExport;
