// Round-trip tests: diagram -> PyTorch -> diagram -> identical PyTorch, in both code styles.
const NN = require('./core.js');
let ok = true;
for (const [preset, literalDims] of [['presetResidualMLP', true], ['presetResidualMLP', false], ['presetResidualMLPBlocks', true], ['presetResidualMLPBlocks', false]]) {
  const a = NN.generate(NN[preset](), { literalDims }).code;
  const g = NN.importPython(a).graph;
  const b = NN.generate(g, { literalDims }).code;
  if (a !== b) { ok = false; console.error(`Round trip changed the code (${preset}, literalDims=${literalDims})`); require('fs').writeFileSync(`/tmp/rt_${literalDims}_a.py`, a); require('fs').writeFileSync(`/tmp/rt_${literalDims}_b.py`, b); }
}
if (!ok) process.exit(1);
console.log('round trip OK');
// The block example is one GatedBlock repeated 3 times: nn.ModuleList + loop, and it reads back the same way.
{
  const g = NN.presetResidualMLPBlocks();
  const a = NN.generate(g).code, imp = NN.importPython(a), b = NN.generate(imp.graph).code;
  const use = imp.graph.nodes.find(n => n.type === 'block');
  if (a !== b || !/self\.blocks = nn\.ModuleList\(\[GatedBlock\(256, 512\) for _ in range\(3\)\]\)/.test(a) || +use.params.repeat !== 3) { console.error('Repeated block round trip failed'); process.exit(1); }
  console.log('repeated block round trip OK');
}
// The fully expanded transformer example round-trips in both styles.
{
  const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/06_decoder_transformer.json', 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, b = NN.generate(NN.importPython(a).graph, { literalDims }).code;
    if (a !== b) { console.error('Transformer round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  console.log('transformer round trip OK');
}
// GPT-2: one GPT2Block repeated 12 times, round-trips in both styles as a single repeated block.
{
  const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/07_gpt2.json', 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, imp = NN.importPython(a), b = NN.generate(imp.graph, { literalDims }).code;
    const use = imp.graph.nodes.find(n => n.type === 'block');
    if (a !== b || !use || +use.params.repeat !== 12) { console.error('GPT-2 round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  console.log('GPT-2 round trip OK');
}
// BERT-base, Llama 2 7B and Qwen2.5-VL 3B: round trips and exact parameter counts.
for (const [f, want] of [['08_bert_base.json', 109514298], ['09_llama2_7b.json', 6738415616], ['10_qwen2_5_vl_3b.json', 3754622976]]) {
  const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) { const a = NN.generate(g, { literalDims }).code, b = NN.generate(NN.importPython(a).graph, { literalDims }).code; if (a !== b) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); } }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== want) { console.error(f + ' has ' + n + ' parameters, expected ' + want); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters');
}
// Losses: loss() and predict() are written and read back for every kind.
{
  const fs = require('fs');
  const dir = __dirname + '/test_fixtures/losses/';
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.py'))) {
    const a = fs.readFileSync(dir + f, 'utf8'), g = NN.importPython(a).graph, b = NN.generate(g).code;
    if (a !== b || !g.nodes.some(n => n.type === 'loss')) { console.error('loss round trip failed: ' + f); process.exit(1); }
  }
  const X = require('./export.js');
  for (const f of ['06_decoder_transformer.json', '07_gpt2.json', '08_bert_base.json', '09_llama2_7b.json']) {
    const g = JSON.parse(fs.readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph, r = X.trainScript(g, {});
    if (r.errors.length || !/def compute_loss/.test(r.train)) { console.error('export failed: ' + f + ' ' + r.errors); process.exit(1); }
    // only the selections are written into the code
    const t = X.trainScript(g, { optimizer: 'sgd', schedule: 'constant', warmup: 0, precision: 'fp32', grad_clip: 0 }).train;
    if (/AdamW|lr_scheduler|autocast|argparse|clip_grad|wandb|RESUME/.test(t) || !/torch\.optim\.SGD/.test(t)) { console.error('export not specialised: ' + f); process.exit(1); }
  }
  console.log('loss round trips and exports OK');
  // blocks with several inputs and outputs: one class per block, stable round trip
  const md = __dirname + '/test_fixtures/multi/';
  for (const f of fs.readdirSync(md).filter(f => f.endsWith('.py'))) {
    const g = NN.importPython(fs.readFileSync(md + f, 'utf8')).graph, code = NN.generate(g).code;
    const multi = Object.values(g.blocks || {}).some(b => NN.blockOutputs(b).length > 1);
    if (!multi || Object.keys(g.blocks).length !== 1 || (code.match(/^class /gm) || []).length !== 2 || NN.generate(NN.importPython(code).graph).code !== code) { console.error('multi-output round trip failed: ' + f); process.exit(1); }
  }
  console.log('multi-output blocks OK');
  // blocks inside blocks: nested classes come back as nested blocks, sizes pass through, one class per block
  const nd = __dirname + '/test_fixtures/nested/';
  for (const f of fs.readdirSync(nd).filter(f => f.endsWith('.py'))) {
    const g = NN.importPython(fs.readFileSync(nd + f, 'utf8')).graph, code = NN.generate(g).code;
    const nested = Object.values(g.blocks || {}).some(b => b.graph.nodes.some(n => n.type === 'block'));
    const names = (code.match(/^class (\w+)/gm) || []);
    if (!nested || new Set(names).size !== names.length || NN.generate(NN.importPython(code).graph).code !== code) { console.error('nested blocks round trip failed: ' + f); process.exit(1); }
  }
  console.log('nested blocks OK');
}
// Source sync: a diagram edit is written into the file text; everything the edit does not touch stays byte for byte.
{
  const fs = require('fs');
  const check = (cond, msg) => { if (!cond) { console.error('Source sync failed: ' + msg); process.exit(1); } };
  const parse = text => NN.importPython(text, {}).graph;
  const dir = __dirname + '/test_fixtures/';
  const nb = fs.readFileSync(dir + 'marimo/marimo-net.py', 'utf8');
  const plain = NN.generate(NN.presetResidualMLP()).code + '\n# written by hand\n';
  check(NN.isMarimo(nb) && !NN.isMarimo(plain), 'marimo detection');
  // An unchanged graph returns the text exactly.
  for (const src of [nb, plain]) check(NN.syncSource(src, parse(src), {}) === src, 'unchanged graph rewrote the file');
  // A size change rewrites only the statement it touches.
  const sized = parse(nb);
  sized.nodes.find(n => n.params.name === 'fc1').params.out = '32';
  check(NN.syncSource(nb, sized, {}) === nb.replace('nn.Linear(n_in, 16)', 'nn.Linear(n_in, 32)'), 'size change touched other lines');
  // A new layer is added to __init__ and forward is rewritten; comments, docstrings and other methods stay.
  const edited = nb
    .replace('        self.act = nn.ReLU()\n', '        self.act = nn.ReLU()\n        self.drop = nn.Dropout(0.1)\n')
    .replace('self.fc2(self.act(self.fc1(x)))', 'self.fc2(self.drop(self.act(self.fc1(x))))');
  const grown = NN.syncSource(nb, parse(edited), {});
  check(/self\.drop = nn\.Dropout\(0\.1\)/.test(grown), 'new layer missing');
  for (const kept of ['# keep me', '"""Doc."""', 'def predict', '@app.class_definition', 'n_in, 16)']) check(grown.includes(kept), 'lost ' + kept);
  check(NN.syncSource(grown, parse(edited), {}) === grown, 'syncing twice changed the file');
  // Renaming the model renames its class and nothing else.
  const renamed = parse(nb); renamed.name = 'Renamed';
  check(NN.syncSource(nb, renamed, {}) === nb.replace('class Net(nn.Module):', 'class Renamed(nn.Module):'), 'rename changed other lines or deleted the class');
  // A block the diagram removes loses its class and its use; the other classes stay.
  const enc = fs.readFileSync(dir + 'nested/encoder.py', 'utf8');
  const eg = parse(enc), B = eg.blocks;
  const mixId = Object.keys(B).find(k => B[k].name === 'Mix');
  const encG = B[Object.keys(B).find(k => B[k].name === 'Encoder')].graph;
  const use = encG.nodes.find(n => n.type === 'block' && n.params.block === mixId);
  const inE = encG.edges.find(e => e.to === use.id), outE = encG.edges.find(e => e.from === use.id);
  encG.nodes = encG.nodes.filter(n => n !== use);
  encG.edges = encG.edges.filter(e => e !== inE && e !== outE).concat([{ id: 'bridge', from: inE.from, to: outE.to, port: outE.port }]);
  delete B[mixId];
  const del = NN.syncSource(enc, eg, {});
  check(!/^class Mix\b/m.test(del) && /^class Gate\b/m.test(del) && /^class Encoder\b/m.test(del), 'removed class not deleted, or others deleted');
  check(!/self\.mix/.test(del) && !/\n{4,}/.test(del), 'removed block left its use or extra blank lines');
  // A new helper function gets @app.function, and a missing import goes into the setup cell.
  const masked = parse(nb), out = masked.nodes.find(n => n.type === 'output'), fc2 = masked.nodes.find(n => n.params.name === 'fc2');
  const mk = { id: 'mk1', type: 'mask', params: { mode: 'causal', title: '', var: '' }, x: 0, y: 0 };
  masked.nodes.push(mk);
  masked.edges = masked.edges.map(e => (e.to === out.id ? Object.assign({}, e, { from: mk.id }) : e)).concat([{ id: 'ex1', from: fc2.id, to: mk.id, port: 0 }]);
  const mo = NN.syncSource(nb, masked, {});
  check(/^@app\.function\ndef causal_mask\(scores\):/m.test(mo) && /causal_mask\(self\.fc2/.test(mo), 'helper not decorated');
  check(/^    import torch\.nn\.functional as F$/m.test(mo) && !/^import torch\.nn\.functional/m.test(mo), 'import not in setup cell');
  // Merging generated classes into a notebook keeps its decorators and setup imports.
  const gen = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n\n    def forward(self, x):\n        return F.relu(self.fc(x))\n\n\nclass Extra(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(2, 2)\n\n    def forward(self, x):\n        return self.fc(x)\n';
  const merged = NN.mergeIntoSource(nb, gen, NN.importPython(nb, {}).usedClasses);
  check(/^@app\.class_definition\nclass Net\(nn\.Module\):/m.test(merged) && /^@app\.class_definition\nclass Extra\b/m.test(merged), 'merged classes not decorated');
  check(/^    import torch\.nn\.functional as F$/m.test(merged) && !/^import torch\.nn\.functional/m.test(merged) && !/^import torch\.nn as nn/m.test(merged), 'merged imports not in setup cell');
  // A new block with its own class is inserted before the class that uses it, decorated in a marimo notebook.
  const withBlock = parse(nb), bOut = withBlock.nodes.find(n => n.type === 'output'), fc2b = withBlock.nodes.find(n => n.params.name === 'fc2');
  const sq = { name: 'Squash', symbols: { e: 8, d: 4 }, nodes: [], edges: [] };
  const SB = (type, params) => { const n = NN.makeNode(type, params); sq.nodes.push(n); return n; };
  const sx = SB('input', { name: 'x', shape: 'e, d' }), sa = SB('act', { fn: 'tanh' }), so = SB('output', { name: 'out' });
  sq.edges.push({ id: 'sq1', from: sx.id, to: sa.id, port: 0 }, { id: 'sq2', from: sa.id, to: so.id, port: 0 });
  withBlock.blocks = Object.assign({}, withBlock.blocks, { sqBlk: { id: 'sqBlk', name: 'Squash', graph: sq } });
  const sqUse = NN.makeNode('block', { block: 'sqBlk', args: {}, nIn: 1, repeat: 1, name: 'squash' });
  withBlock.nodes.push(sqUse);
  withBlock.edges = withBlock.edges.filter(e => e.to !== bOut.id).concat([{ id: 'sq3', from: fc2b.id, to: sqUse.id, port: 0 }, { id: 'sq4', from: sqUse.id, to: bOut.id, port: 0 }]);
  const withNew = NN.syncSource(nb, withBlock, {});
  check((withNew.match(/^class Squash\b/gm) || []).length === 1, 'new class not written exactly once');
  check(/^@app\.class_definition\nclass Squash\(nn\.Module\):/m.test(withNew), 'new class not decorated');
  check(withNew.indexOf('class Squash') < withNew.indexOf('class Net'), 'new class not placed before its user');
  check(withNew.includes('    """Tiny net."""\n\n    def __init__(self, n_in=8, n_out=4):\n        super().__init__()\n        # keep me\n        self.fc1 = nn.Linear(n_in, 16)\n') && withNew.endsWith(nb.slice(nb.indexOf('    def predict'))), 'text outside the change was rewritten');
  check(NN.syncSource(withNew, withBlock, {}) === withNew, 'syncing the new class twice changed the file');
  console.log('source sync OK');
}
// A literal file: a size set in the diagram (through its symbol) reaches the file, as the inspector's Output size does.
{
  const base = NN.generate(NN.presetResidualMLP(), { literalDims: true }).code;
  const g = NN.importPython(base).graph;
  g.symbols.d256 = 64;
  NN.syncWidths(g);
  const out = NN.syncSource(base, g, { gen: { literalDims: true, shapeComments: true } });
  if (out === base || !/nn\.Linear\(784, 64, bias=False\)/.test(out)) { console.error('A symbol size edit did not reach a literal file'); process.exit(1); }
  console.log('literal file symbol size OK');
}
