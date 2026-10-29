// Writes every example network as runnable files, for tools/check_examples.py:
//   <out>/<name>/model.py and model_named.py (the code panel's code, sizes as numbers and as arguments),
//   <out>/<name>/train.py (a tiny train.py: 2 steps on synthetic data, then resume), and <out>/manifest.json
//   (the parameter count the app shows). usage: node tools/export_examples.js out/
const fs = require('fs'), path = require('path');
const NN = require('../core.js'), X = require('../export.js');
const out = process.argv[2]; if (!out) { console.error('usage: node tools/export_examples.js out/'); process.exit(2); }
const ex = path.join(__dirname, '..', 'examples', 'networks'), man = [];
for (const f of fs.readdirSync(ex).sort()) {
  const src = fs.readFileSync(path.join(ex, f), 'utf8'), name = f.replace(/\.(json|py)$/, '');
  const g = f.endsWith('.json') ? JSON.parse(src).graph : NN.importPython(src).graph;
  const dir = path.join(out, name); fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'model.py'), NN.generate(g, { literalDims: true }).code);
  fs.writeFileSync(path.join(dir, 'model_named.py'), NN.generate(g, { literalDims: false }).code);
  const fl = NN.expandBlocks(g).flat; let params = 0; for (const n of fl.nodes) params += NN.paramCount(n, fl.symbols) || 0;
  const t = X.trainScript(g, { length: 'steps', max_steps: 2, num_samples: 16, batch_size: 4, val_fraction: 0.25, log_every: 1, eval_every: 2, save_every: 2, warmup: 1, out_dir: 'ckpt', device: 'cpu', grad_accum: 1, resume: true });
  const tb = X.trainScript(g, { length: 'steps', max_steps: 1, num_samples: 8, batch_size: 4, val_fraction: 0, log_every: 1, eval_every: 0, save_every: 0, out_dir: 'ckpt_bf16', device: 'cpu', precision: 'bf16', optimizer: 'sgd', schedule: 'onecycle', ema: 0.99, grad_accum: 2, grad_clip: 1 });
  if (!t.errors || !t.errors.length) { fs.writeFileSync(path.join(dir, 'train.py'), t.train); fs.writeFileSync(path.join(dir, 'train_bf16.py'), tb.train); }
  man.push({ name, params, train: !(t.errors && t.errors.length), trainErrors: t.errors || [] });
}
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(man, null, 1));
console.log(`wrote ${man.length} examples to ${out}`);
