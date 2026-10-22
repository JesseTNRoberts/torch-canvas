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
