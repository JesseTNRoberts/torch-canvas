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
// A graph from a link or a paste is checked: a setting with a fixed set of choices that isn't one of them goes back to its default
{
  const NN = require('./core.js');
  const g = { name: 'G', nodes: [{ id: 'a', type: 'act', params: { fn: '<b>x</b>' } }, { id: 'b', type: 'binop', params: { op: '<img>' } }, { id: 'c', type: 'linear', params: { init: 'nope' } }, { id: 'd', type: 'norm', params: { kind: 'LayerNorm' } }, { id: 'e', type: 'act', params: { fn: 'gelu' } }], edges: [] };
  const s = NN.sanitizeGraph(g), p = id => s.nodes.find(n => n.id === id).params;
  if (p('a').fn !== 'silu' || p('b').op !== '+' || p('c').init !== 'default' || p('d').kind !== 'LayerNorm' || p('e').fn !== 'gelu') { console.error('sanitizeGraph: settings outside their choices were kept', p('a').fn, p('b').op, p('c').init); process.exit(1); }
  console.log('graph settings OK');
}
// BERT-base, Llama 2 7B and Qwen2.5-VL 3B: round trips and exact parameter counts.
for (const [f, want] of [['08_bert_base.json', 109514298], ['09_llama2_7b.json', 6738415616], ['10_qwen2_5_vl_3b.json', 3754622976]]) {
  const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) { const a = NN.generate(g, { literalDims }).code, b = NN.generate(NN.importPython(a).graph, { literalDims }).code; if (a !== b) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); } }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== want) { console.error(f + ' has ' + n + ' parameters, expected ' + want); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters');
}
// Image models: a plain CNN, ResNet-50 (bottleneck blocks) and YOLOv3 (nested Darknet blocks, three outputs).
// Exact parameter counts (torchvision's resnet50, Darknet's yolov3.cfg); the code reads back to the same code, and the
// named style (sizes as arguments) reads back to the same model.
for (const [f, want, styles] of [['11_basic_cnn.json', 421642, [true, false]], ['12_resnet50.json', 25557032, [true]], ['13_yolov3.json', 61949149, [true]]]) {
  const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of styles) { const a = NN.generate(g, { literalDims }).code, b = NN.generate(NN.importPython(a).graph, { literalDims }).code; if (a !== b) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); } }
  const lit = NN.generate(g).code, named = NN.generate(g, { literalDims: false }).code;
  if (NN.generate(NN.importPython(named).graph).code !== lit) { console.error(f + ': the named code reads back as a different model'); process.exit(1); }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== want) { console.error(f + ' has ' + n + ' parameters, expected ' + want); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters');
}
// Variance plot on image models: every Linear and Conv2d is plotted, named like PyTorch's modules, and the first
// convolution of the basic CNN matches its closed form: 9 taps · Var(w) = 1/27, times the share of taps inside the
// 28 × 28 map with padding 1, plus the bias variance 1/27.
{
  const V = require('./engine.js');
  for (const [f, n, first] of [['11_basic_cnn.json', 4, 'conv1'], ['12_resnet50.json', 54, 'conv1'], ['13_yolov3.json', 75, 'stem.conv']]) {
    const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph, r = V.predict(g);
    if (r.error || r.layers.length !== n || r.layers[0].name !== first || r.layers.some(l => !(l.fwdOut > 0 && l.bwdIn > 0))) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
    if (f === '11_basic_cnn.json') { const share = ((28 * 3 - 2) / (28 * 3)) ** 2, want = 9 / 27 * share + 1 / 27; if (Math.abs(r.layers[0].fwdOut / want - 1) > 0.01) { console.error('conv variance ' + r.layers[0].fwdOut + ', expected ' + want); process.exit(1); } }
  }
  console.log('variance plot on image models OK');
}
// Recurrences: RNN, LSTM and GRU drawn flat with Delays. The code is a loop over time that reads back into the same
// diagram (Delays included), with nn.RNN / nn.LSTM / nn.GRU's parameter counts.
for (const [f, want, delays] of [['14_rnn.json', 5768, 1], ['15_lstm.json', 21512, 2], ['16_gru.json', 16264, 1]]) {
  const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, imp = NN.importPython(a).graph, b = NN.generate(imp, { literalDims }).code;
    if (a !== b || !/for t in range\(/.test(a) || imp.nodes.filter(n => n.type === 'delay').length !== delays) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  let n = 0; for (const z of g.nodes) n += NN.paramCount(z, g.symbols) || 0;
  if (n !== want) { console.error(f + ' has ' + n + ' parameters, expected ' + want); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters');
  // the variance plot looks at one step, with the state it settles to; the Elman RNN's hidden-to-hidden layer reads
  // tanh of a sum whose variance is what that state gives back (a fixed point), so its input variance stays well inside (0, 1)
  const r = require('./engine.js').predict(g);
  if (r.error || r.layers.length !== { 1: 3, 2: 9 }[delays] && f !== '16_gru.json' || r.layers.some(l => !(l.fwdOut > 0 && l.bwdIn > 0)) || !/one step/.test(r.notes[0])) { console.error('variance plot failed: ' + f + ' ' + (r.error || '')); process.exit(1); }
}
// A spiking layer: leaky integrate-and-fire neurons drawn flat with two Delays (the membrane potential and the last
// spike, which resets it). Spike is an
// activation: a step forward, written with a helper that gives it a surrogate gradient. Reads back in both styles
// with the surrogate slope kept, and a slope other than the default is written out.
{
  const f = '26_lif.json', g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, imp = NN.importPython(a).graph, b = NN.generate(imp, { literalDims }).code;
    const sp = imp.nodes.filter(n => n.type === 'act' && n.params.fn === 'spike');
    if (a !== b || !/for t in range\(/.test(a) || !/^def spike\(x, slope=25\.0\):/m.test(a) || !/v = cur\[:, t\] \+ v \* 0\.9 - s \* 1\b/.test(a) || !/s = spike\(v - 1\)/.test(a) || imp.nodes.filter(n => n.type === 'delay').length !== 2 || sp.length !== 1 || +sp[0].params.slope !== 25) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  let n = 0; for (const z of g.nodes) n += NN.paramCount(z, g.symbols) || 0;
  if (n !== 1024) { console.error(f + ' has ' + n + ' parameters, expected 1024'); process.exit(1); }
  const g2 = JSON.parse(JSON.stringify(g)); g2.nodes.find(z => z.params.fn === 'spike').params.slope = '10';
  const c2 = NN.generate(g2).code, back = NN.importPython(c2).graph.nodes.find(z => z.params.fn === 'spike');
  if (!/spike\(v - 1, 10\)/.test(c2) || !back || +back.params.slope !== 10) { console.error(f + ': a surrogate slope of 10 was not kept'); process.exit(1); }
  // a model without a Spike doesn't get the helper
  if (/def spike\(/.test(NN.generate(JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/14_rnn.json', 'utf8')).graph).code)) { console.error('spike helper written for a model without one'); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters');
}
// Vision Transformer (torchvision's vit_b_16): patches by a strided convolution, flattened into a sequence, a class
// token in front, position embeddings, 12 encoder blocks, the class token's position into the head. Same parameter
// count as torchvision; reads back in both styles; the variance plot simulates every patch as its own position.
{
  const f = '17_vit_b_16.json', g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, imp = NN.importPython(a).graph, b = NN.generate(imp, { literalDims }).code;
    if (a !== b || !imp.nodes.some(n => n.type === 'token') || !imp.nodes.some(n => n.type === 'select')) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  if (NN.generate(NN.importPython(NN.generate(g, { literalDims: false }).code).graph).code !== NN.generate(g).code) { console.error(f + ': the named code reads back as a different model'); process.exit(1); }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== 86567656) { console.error(f + ' has ' + n + ' parameters, expected 86567656'); process.exit(1); }
  const r = require('./engine.js').predict(g);
  if (r.error || r.layers.length !== 74 || r.layers[0].name !== 'conv_proj' || r.layers[73].name !== 'head' || r.layers.some(l => !(l.fwdOut > 0 && l.bwdIn > 0))) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters, variance plot OK');
}
// Mamba-130M: the selective scan drawn flat with a Delay (a loop of elementwise ops, run as a for loop over time),
// with a learned A_log (S4D-real init) and D, a causal depthwise Conv1d, softplus and Mamba's Δ bias init. Same
// parameter count as transformers' MambaForCausalLM; reads back in both styles; the variance plot runs the scan
// along the positions (no settling: the loop has no weights).
{
  const f = '18_mamba_130m.json', g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, imp = NN.importPython(a), b = NN.generate(imp.graph, { literalDims }).code;
    const inner = Object.values(imp.graph.blocks || {})[0], kinds = inner ? inner.graph.nodes.map(n => n.type) : [];
    if (a !== b || !/for t in range\(/.test(a) || !/dt_bias_\(self\.dt_proj\.bias\)/.test(a) || kinds.filter(k => k === 'param').length !== 2 || !kinds.includes('conv1d') || !kinds.includes('delay')) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  if (NN.generate(NN.importPython(NN.generate(g, { literalDims: false }).code).graph).code !== NN.generate(g).code) { console.error(f + ': the named code reads back as a different model'); process.exit(1); }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== 129135360) { console.error(f + ' has ' + n + ' parameters, expected 129135360'); process.exit(1); }
  const r = require('./engine.js').predict(g);
  if (r.error || r.layers.length !== 193 || r.layers[2].name !== 'layers.0.conv1d' || r.layers.some(l => !(l.fwdOut > 0 && l.bwdIn > 0)) || r.notes.some(x => /one step/.test(x))) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters, variance plot OK');
}
// Residual CNNs (pre-norm blocks with image GroupNorm): the global-average-pooling head and the dense head share one
// block; both round-trip, have the right parameter counts, and plot.
for (const [f, want] of [['23_residual_cnn_gap.json', 57098], ['24_residual_cnn_dense.json', 72458], ['25_residual_cnn_expanded.json', 57098]]) {
  const g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  const a = NN.generate(g, { literalDims: true }).code, b = NN.generate(NN.importPython(a).graph, { literalDims: true }).code;
  if (a !== b || !/nn\.GroupNorm\(1, 32\)/.test(a) || /GroupNorm\(1, 7\)/.test(a)) { console.error(f + ' round trip failed'); process.exit(1); }
  const fl = NN.expandBlocks(g).flat; let n = 0; for (const z of fl.nodes) n += NN.paramCount(z, fl.symbols) || 0;
  if (n !== want) { console.error(f + ' has ' + n + ' parameters, expected ' + want); process.exit(1); }
  const r = require('./engine.js').predict(g);
  if (r.error || r.layers.length !== 8) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters, variance plot OK');
}
// VAE (PyTorch's examples/vae): three outputs (reconstruction logits, μ, log σ²), a Noise node for the
// reparameterization trick, and a loss of 784 × BCE plus the KL term (kl_normal), read back as the same losses.
{
  const f = '19_vae_mnist.json', g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, imp = NN.importPython(a).graph, b = NN.generate(imp, { literalDims }).code;
    if (a !== b || !/torch\.randn_like\(std\)/.test(a) || !/kl_normal\(mu, logvar\)/.test(a) || imp.nodes.filter(n => n.type === 'loss').length !== 2 || !imp.nodes.some(n => n.type === 'noise')) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  let n = 0; for (const z of g.nodes) n += NN.paramCount(z, g.symbols) || 0;
  if (n !== 652824) { console.error(f + ' has ' + n + ' parameters, expected 652824'); process.exit(1); }
  const r = require('./engine.js').predict(g);
  if (r.error || r.layers.length !== 5 || r.layers.some(l => !(l.fwdOut > 0 && l.bwdIn > 0))) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters, variance plot OK');
}
// U-Net (milesial/Pytorch-UNet, bilinear=False): DoubleConv, Down (nested) and Up (two inputs: the deeper map and
// the skip connection) blocks, ConvTranspose2d on the way up, per-pixel cross-entropy. Same parameters as the reference.
{
  const f = '20_unet.json', g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  const a = NN.generate(g).code, imp = NN.importPython(a).graph, b = NN.generate(imp).code;
  if (a !== b || !/nn\.ConvTranspose2d\(c_in, c_out, 2, stride=2\)/.test(a) || !/F\.cross_entropy\(logits, mask\)/.test(a) || !/torch\.softmax\(self\(x\), dim=1\)/.test(a)) { console.error(f + ' round trip failed'); process.exit(1); }
  if (NN.generate(NN.importPython(NN.generate(g, { literalDims: false }).code).graph).code !== a) { console.error(f + ': the named code reads back as a different model'); process.exit(1); }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== 31037698) { console.error(f + ' has ' + n + ' parameters, expected 31037698'); process.exit(1); }
  const r = require('./engine.js').predict(g);
  if (r.error || r.layers.length !== 23 || r.layers.filter(l => /\.up$/.test(l.name)).length !== 4 || r.layers.some(l => !(l.fwdOut > 0 && l.bwdIn > 0))) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters, variance plot OK');
}
// The original Transformer (encoder-decoder, as nn.Transformer): EncoderLayer × 6, DecoderLayer × 6 with two inputs
// (the decoder's sequence and the encoder's memory, cross-attention), source length s and target length t, a shared
// embedding scaled by √d (d ** 0.5, read back as written), sinusoidal positions and a tied output projection.
{
  const f = '21_transformer_base.json', g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, b = NN.generate(NN.importPython(a).graph, { literalDims }).code;
    if (a !== b || !/def forward\(self, x, memory\)/.test(a) || !/ = \w+ \+ sinusoidal_pe\(\w+\)/.test(a) || !(literalDims ? /\* 512 \*\* 0\.5/ : /\* self\.d \*\* 0\.5/).test(a) || !/# \(e=8, s=64\)/.test(a)) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  if (NN.generate(NN.importPython(NN.generate(g, { literalDims: false }).code).graph).code !== NN.generate(g).code) { console.error(f + ': the named code reads back as a different model'); process.exit(1); }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== 63084544) { console.error(f + ' has ' + n + ' parameters, expected 63084544'); process.exit(1); }
  const r = require('./engine.js').predict(g);
  if (r.error || r.layers.length !== 97 || r.layers.filter(l => /\.ca_q$/.test(l.name)).length !== 6 || r.layers.some(l => !(l.fwdOut > 0 && l.bwdIn > 0))) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters, variance plot OK');
}
// Freezing: a layer, a parameter, a class token, a causal conv, a block use (ModuleList) and a layer inside a block are
// written as requires_grad_(False), read back frozen, and left out of the exported optimizer.
{
  const fs = require('fs'), X = require('./export.js');
  const cases = [
    ['17_vit_b_16.json', g => { g.nodes.find(n => n.type === 'token').params.frozen = true; g.nodes.find(n => n.type === 'conv').params.frozen = true; g.nodes.find(n => n.type === 'block').params.frozen = true; }, 3],
    ['18_mamba_130m.json', g => { const B = Object.values(g.blocks)[0]; B.graph.nodes.find(n => n.params.name === 'A_log').params.frozen = true; B.graph.nodes.find(n => n.type === 'conv1d').params.frozen = true; }, 2],
    ['20_unet.json', g => { g.nodes.find(n => n.params.name === 'inc').params.frozen = true; g.nodes.find(n => n.params.name === 'outc').params.frozen = true; }, 2],
  ];
  for (const [f, pick, want] of cases) {
    const g = JSON.parse(fs.readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph; pick(g);
    const a = NN.generate(g).code, b = NN.generate(NN.importPython(a).graph).code;
    if (a !== b || (a.match(/requires_grad_\(False\)/g) || []).length !== want) { console.error('freeze round trip failed: ' + f); process.exit(1); }
    if (!/p\.requires_grad/.test(X.trainScript(g, {}).train)) { console.error('freeze: the optimizer still gets every weight: ' + f); process.exit(1); }
  }
  console.log('frozen layers and blocks OK');
}
// Small I-JEPA: the target encoder (a block use) and its patch embedding (a layer) follow the context ones (EMA ties:
// copied, frozen, run under no_grad, moved by ema_update with momentum 0.996 → 1.0); index inputs and take_positions /
// last_positions; fixed 2-D sin-cos positions (frozen Parameters); a LayerNorm without weights; a loss between two
// outputs. Reads back in both styles; the variance plot gives the followers no gradients; train.py draws the masks.
{
  const f = '22_ijepa_small.json', g = JSON.parse(require('fs').readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph;
  for (const literalDims of [true, false]) {
    const a = NN.generate(g, { literalDims }).code, imp = NN.importPython(a).graph, b = NN.generate(imp, { literalDims }).code;
    const fol = imp.nodes.filter(n => n.params.follows);
    if (a !== b || fol.length !== 2 || fol.some(n => n.params.momentum !== '0.996' || n.params.momentum_end !== '1.0') || (a.match(/with torch\.no_grad\(\):/g) || []).length !== 2 || !/def ema_update\(self, progress\)/.test(a) || !/F\.huber_loss\(pred, targets/.test(a)) { console.error(f + ' round trip failed, literalDims=' + literalDims); process.exit(1); }
  }
  if (NN.generate(NN.importPython(NN.generate(g, { literalDims: false }).code).graph).code !== NN.generate(g).code) { console.error(f + ': the named code reads back as a different model'); process.exit(1); }
  const flat = NN.expandBlocks(g).flat; let n = 0; for (const z of flat.nodes) n += NN.paramCount(z, flat.symbols) || 0;
  if (n !== 5749344) { console.error(f + ' has ' + n + ' parameters, expected 5749344'); process.exit(1); }
  const r = require('./engine.js').predict(g), tgt = r.layers && r.layers.filter(l => /^target_/.test(l.name));
  if (r.error || r.layers.length !== 94 || tgt.length !== 37 || tgt.some(l => l.bwdIn != null) || r.layers.filter(l => !/^target_/.test(l.name)).some(l => !(l.fwdOut > 0 && l.bwdIn > 0))) { console.error('variance plot failed: ' + f + ' ' + (r.error || r.layers.length)); process.exit(1); }
  const t = require('./export.js').trainScript(g, {}).train;
  if (!/ctx_idx, tgt_idx = sample_masks\(imgs\.size\(0\), 64, imgs\.device\)/.test(t) || !/model\.ema_update\(\(step \+ 1\) \/ total_steps\)/.test(t)) { console.error('export failed: ' + f); process.exit(1); }
  console.log(f + ' round trip OK, ' + n.toLocaleString('en-US') + ' parameters, variance plot and export OK');
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
  for (const f of ['06_decoder_transformer.json', '07_gpt2.json', '08_bert_base.json', '09_llama2_7b.json', '11_basic_cnn.json', '12_resnet50.json', '17_vit_b_16.json', '18_mamba_130m.json', '19_vae_mnist.json', '20_unet.json', '21_transformer_base.json']) {
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
  // The reader keeps, for every node, the lines of the statement it was read from (the innermost one) and the
  // self.<name> of the module it is, so the code panel can highlight a component in a file's own text.
  {
    const src = ['import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '', '',
      'class Inner(nn.Module):', '    def __init__(self, d=16):', '        super().__init__()', '        self.lin = nn.Linear(d, d)', '',
      '    def forward(self, x):', '        y = self.lin(x)', '        return torch.relu(y)', '', '',
      'class Net(nn.Module):', '    def __init__(self):', '        super().__init__()', '        self.fc1 = nn.Linear(8, 16)', '        self.act = nn.ReLU()', '        self.inner = Inner(16)', '',
      '    def forward(self, x):', '        h = self.act(self.fc1(x))', '        h = h + self.inner(h)', '        h = F.gelu(', '            h', '        )', '        return h', ''].join('\n');
    const r = NN.importPython(src), g = r.graph, at = n => (r.lines.get(n.id) || []).join(','), bad = m => { console.error('source lines failed: ' + m); process.exit(1); };
    const find = (gr, f) => gr.nodes.find(f), blk = Object.values(g.blocks)[0].graph;
    if (at(find(g, n => n.type === 'linear')) !== '24' || r.attrs.get(find(g, n => n.type === 'linear').id) !== 'fc1') bad('a layer');
    if (at(find(g, n => n.params.fn === 'relu')) !== '24' || r.attrs.get(find(g, n => n.params.fn === 'relu').id) !== 'act') bad('an activation written as a module');
    if (at(find(g, n => n.params.fn === 'gelu')) !== '26,27,28') bad('a statement over several lines');
    if (at(find(g, n => n.type === 'binop')) !== '25' || at(find(g, n => n.type === 'block')) !== '25') bad('a block use is the line that calls it');
    if (at(find(g, n => n.type === 'input')) !== '23' || at(find(g, n => n.type === 'output')) !== '29') bad('input and output arrows');
    if (at(find(blk, n => n.type === 'input')) !== '11' || at(find(blk, n => n.type === 'output')) !== '13') bad("a block's own input and output arrows: its forward() and its return");
    if (at(find(blk, n => n.type === 'linear')) !== '12' || at(find(blk, n => n.type === 'act')) !== '13' || r.attrs.get(find(blk, n => n.type === 'linear').id) !== 'lin') bad('layers inside a block keep the lines of its own forward');
    { // a Delay: where its variable starts, the loop, the lines that read the earlier step; a loss: the return of loss()
      const code = NN.generate(JSON.parse(fs.readFileSync(__dirname + '/examples/networks/15_lstm.json', 'utf8')).graph, {}).code, L = code.split('\n'), imp = NN.importPython(code);
      const text = n => (imp.lines.get(n.id) || []).map(l => L[l - 1].trim().replace(/\s+#.*$/, '').split(' ')[0]).join(' ');
      const c = imp.graph.nodes.find(n => n.type === 'delay' && n.params.name === 'c'), loss = imp.graph.nodes.find(n => n.type === 'loss'), tgt = imp.graph.nodes.find(n => n.type === 'input' && n.params.name === 'target');
      if (text(c) !== 'c for c' || text(loss) !== 'return' || text(tgt) !== 'def') bad('Delay, loss or target lines: ' + [text(c), text(loss), text(tgt)].join(' / '));
    }
    // on the code the app writes, a node's source lines are among the lines the generator gives it
    for (const f of ['01_residual_mlp.json', '07_gpt2.json', '15_lstm.json', '12_resnet50.json']) {
      const code = NN.generate(JSON.parse(fs.readFileSync(__dirname + '/examples/networks/' + f, 'utf8')).graph, {}).code, imp = NN.importPython(code), gen = NN.generate(imp.graph, {});
      if (gen.code !== code) bad(f + ' round trip');
      let n = 0;
      for (const gr of [imp.graph, ...Object.values(imp.graph.blocks || {}).map(b => b.graph)]) for (const nd of gr.nodes) {
        const ls = imp.lines.get(nd.id) || [], gl = gen.lineMap.get(nd.id) || []; if (!ls.length || !gl.length || nd.type === 'input') continue;
        n++; if (!ls.every(l => gl.includes(l - 1))) bad(`${f}: ${nd.type} ${nd.params.name || ''} read from line ${ls}, generated on ${gl.map(x => x + 1)}`);
      }
      if (n < 5) bad(f + ': too few nodes with lines');
    }
    { // saving a size change also updates the example input under __main__ when that line is the app's; other lines there stay
      const file = ['import torch', 'import torch.nn as nn', '', '', 'class Net(nn.Module):', '    def __init__(self, d=16):', '        super().__init__()', '        self.fc = nn.Linear(d, 4)  # mine', '',
        '    def forward(self, x):', '        return self.fc(x)', '', '', 'if __name__ == "__main__":', '    torch.manual_seed(0)  # mine', '    model = Net()', '    x = torch.randn(8, 16)  # my batch',
        '    out = model(x)', '    print(out.shape)  # expected torch.Size([8, 4])', ''].join('\n');
      const imp = NN.importPython(file), g2 = JSON.parse(JSON.stringify(imp.graph)); g2.symbols.d = 32;
      const out = NN.syncSource(file, g2, { gen: { literalDims: false } });
      if (!/def __init__\(self, d=32\):/.test(out) || !/    x = torch\.randn\(8, 32\)  # my batch\n/.test(out) || !/torch\.manual_seed\(0\)  # mine/.test(out) || !/# mine\n\n    def forward/.test(out)) bad('example input under __main__ not updated, or other lines changed:\n' + out);
      if (NN.importPython(out).graph.symbols.d !== 32 || NN.syncSource(out, g2, { gen: { literalDims: false } }) !== out) bad('the saved file does not read back with the new size');
    }
    console.log('source lines OK');
  }
  // Saving into a marimo notebook keeps it a notebook: classes keep @app.class_definition, an import the new code
  // needs goes into the `with app.setup:` cell (not above `app = marimo.App()`), and a new block's class is decorated.
  // A plain file is saved as before.
  {
    const nb = fs.readFileSync(__dirname + '/test_fixtures/marimo/marimo-net.py', 'utf8'), plain = fs.readFileSync(__dirname + '/test_fixtures/losses/' + fs.readdirSync(__dirname + '/test_fixtures/losses/').filter(f => f.endsWith('.py'))[0], 'utf8');
    const bad = m => { console.error('marimo save failed: ' + m); process.exit(1); };
    if (!NN.isMarimo(nb) || NN.isMarimo(plain)) bad('detection');
    const imp = NN.importPython(nb), g = imp.graph;
    g.nodes.find(n => n.type === 'linear' && n.params.name === 'fc1').params.out = '32'; g.nodes.find(n => n.type === 'linear' && n.params.name === 'fc2').params.in = '32';
    const gen = NN.generate(g, { literalDims: false }).code, out = NN.mergeIntoSource(nb, gen, imp.usedClasses);
    if (!/^@app\.class_definition\nclass Net\(nn\.Module\):/m.test(out)) bad('the class lost its decorator');
    if ((out.match(/@app\.class_definition/g) || []).length !== 1) bad('decorator repeated');
    if (!/^with app\.setup:\n    import torch\n    import torch\.nn as nn\n    import torch\.nn\.functional as F\n/m.test(out)) bad('the new import is not in the setup cell');
    if (out.indexOf('import torch') < out.indexOf('marimo.App(')) bad('an import was put above the app');
    if (!/nn\.Linear\(n_in, 32\)/.test(out) || !/app\.run\(\)/.test(out)) bad('the edit or the rest of the file');
    if (NN.mergeIntoSource(out, gen, imp.usedClasses) !== out) bad('saving twice changed the file');
    const back = NN.importPython(out).graph; if (NN.generate(back, { literalDims: false }).code !== gen) bad('the saved notebook reads back differently');
    // a plain file goes through the old path untouched
    const pi = NN.importPython(plain), pg = NN.generate(pi.graph, { literalDims: false }).code, po = NN.mergeIntoSource(plain, pg, pi.usedClasses);
    if (po == null || /@\w+\.(class_definition|function)|app\.setup/.test(po)) bad('a plain file got marimo markup');
    console.log('marimo save OK');
  }
  // blocks inside blocks: nested classes come back as nested blocks, sizes pass through, one class per block
  const nd = __dirname + '/test_fixtures/nested/';
  for (const f of fs.readdirSync(nd).filter(f => f.endsWith('.py'))) {
    const g = NN.importPython(fs.readFileSync(nd + f, 'utf8')).graph, code = NN.generate(g).code;
    const nested = Object.values(g.blocks || {}).some(b => b.graph.nodes.some(n => n.type === 'block'));
    const names = (code.match(/^class (\w+)/gm) || []);
    if (!nested || new Set(names).size !== names.length || NN.generate(NN.importPython(code).graph).code !== code) { console.error('nested blocks round trip failed: ' + f); process.exit(1); }
  }
  console.log('nested blocks OK');
  // editing the code: the diagram comes back unchanged (layout, nicknames, notes, size names) when the code is read back,
  // and an edit changes only what it touches
  {
    const ex = __dirname + '/examples/networks/', lay = x => NN.autoLayout(x);
    const sig = g => JSON.stringify([g.nodes.map(n => [n.x, n.y, n.params.title || '', n.params.note || '', n.params.split || null]).sort(), Object.values(g.blocks || {}).map(b => [b.name, b.lib || '', b.graph.nodes.map(n => [n.x, n.y, n.params.title || '', n.params.note || '']).sort()]).sort()]);
    for (const f of fs.readdirSync(ex).filter(f => f.endsWith('.json'))) for (const lit of [true, false]) {
      const g = JSON.parse(fs.readFileSync(ex + f, 'utf8')).graph, o = { literalDims: lit, shapeComments: true };
      const h = NN.carryDiagram(g, NN.importPython(NN.generate(g, o).code).graph, lay);
      if (sig(g) !== sig(h) || NN.generate(h, { literalDims: false }).code !== NN.generate(g, { literalDims: false }).code) { console.error(`code edit round trip changed the diagram: ${f} (literalDims=${lit})`); process.exit(1); }
    }
    const g = JSON.parse(fs.readFileSync(ex + '07_gpt2.json', 'utf8')).graph;
    const h = NN.carryDiagram(g, NN.importPython(NN.generate(g, { literalDims: true }).code.replace(/768/g, '512')).graph, lay);
    if (h.symbols.d !== 512 || !/def __init__\(self, vocab=50257, block=1024, d=512, n_head=12/.test(NN.generate(h, { literalDims: false }).code)) { console.error('code edit: changing every 768 should change d'); process.exit(1); }
    const c = JSON.parse(fs.readFileSync(ex + '24_residual_cnn_dense.json', 'utf8')).graph;
    const src = NN.generate(c, { literalDims: true }).code.replace('        x = self.block3(x)', '        x = self.drop(x)\n        x = self.block3(x)').replace('        self.fc = nn.Linear', '        self.drop = nn.Dropout(0.2)\n        self.fc = nn.Linear');
    const k = NN.carryDiagram(c, NN.importPython(src).graph, lay), d = k.nodes.find(n => n.type === 'dropout');
    const overlap = k.nodes.some(n => n !== d && Math.abs(n.x - d.x) < 100 && Math.abs(n.y - d.y) < 50);
    if (!d || overlap || k.nodes.filter(n => n !== d).some(n => !c.nodes.some(m => m.type === n.type && m.x === n.x))) { console.error('code edit: an added layer should get room of its own in the same column'); process.exit(1); }
  }
  console.log('code edits OK');
  // export_graph.py (torch.fx) output, saved from the examples: it reads back with the same parameter count, with
  // convolutions, pools, flatten and torch.cat as real layers, and the code it writes reads back to the same network
  {
    const want = { residual_mlp_block: 1515008, prenorm_residual_mlp: 2308362, basic_cnn: 421642, vae_mnist: 652824, unet: 31037698, residual_cnn_gap: 57098 };
    for (const [f, n0] of Object.entries(want)) {
      const r = NN.importFx(fs.readFileSync(__dirname + '/test_fixtures/fx/' + f + '.json', 'utf8')), g = r.graph;
      const fl = NN.expandBlocks(g).flat; let n = 0; for (const z of fl.nodes) n += NN.paramCount(z, fl.symbols) || 0;
      if (n !== n0) { console.error(`fx import of ${f}: ${n} parameters, expected ${n0}`); process.exit(1); }
      if (r.warnings.some(w => /Conv2d|pool|flatten|gelu/i.test(w))) { console.error(`fx import of ${f} kept a layer it knows as a custom op: ${r.warnings.join('; ')}`); process.exit(1); }
      // its code reads back to the same network (written the same way from then on)
      const code = NN.generate(g).code, g2 = NN.importPython(code).graph, code2 = NN.generate(g2).code;
      const f2 = NN.expandBlocks(g2).flat; let n2 = 0; for (const z of f2.nodes) n2 += NN.paramCount(z, f2.symbols) || 0;
      if (/return None/.test(code) || n2 !== n0 || NN.generate(NN.importPython(code2).graph).code !== code2) { console.error(`fx import of ${f}: its code doesn't read back to the same network`); process.exit(1); }
    }
  }
  console.log('fx imports OK');
  // damaged code (lines lost, text cut short, junk typed in) is refused with a message, never a crash or a hang
  {
    const fails = require('./tools/fuzz_import.js')(400, 7, () => {});
    if (fails.length) { console.error('reading damaged code went wrong:\n' + fails.slice(0, 3).join('\n\n')); process.exit(1); }
  }
  // a module whose forward() calls itself is refused at once, not after rebuilding the model 150 times (a 4 s fuzz case)
  {
    const big = fs.readFileSync(__dirname + '/examples/networks/22_ijepa_small.json', 'utf8'), g = JSON.parse(big).graph;
    const code = NN.generate(g, { literalDims: false }).code.replace(/(\n(\s+)for blk in self\.predictor_blocks:)/, '\n$2outs = self(imgs, ctx_idx, tgt_idx)$1');
    const t0 = Date.now(); let msg = '';
    try { NN.importPython(code); } catch (e) { msg = e.message; }
    if (!/IJEPA\.forward\(\) calls itself/.test(msg) || Date.now() - t0 > 1000) { console.error(`a forward() that calls itself: ${msg || 'read without error'} after ${Date.now() - t0} ms`); process.exit(1); }
  }
  console.log('damaged code OK');
  // Explode blocks: the same network (parameters, code that reads back), every block drawn out or kept by the rule (or a
  // view's choices), nothing overlapping; views put layers back where they were saved, by code name
  {
    const ex = __dirname + '/examples/networks/', cnt = g => { const fl = NN.expandBlocks(g).flat; let n = 0; for (const z of fl.nodes) n += NN.paramCount(z, fl.symbols) || 0; return n; };
    const overlaps = g => { let c = 0; for (let i = 0; i < g.nodes.length; i++) for (let j = i + 1; j < g.nodes.length; j++) { const a = g.nodes[i], b = g.nodes[j], sa = NN.nodeSize(a), sb = NN.nodeSize(b); if (Math.abs(a.x - b.x) < (sa.w + sb.w) / 2 - 2 && Math.abs(a.y - b.y) < (sa.h + sb.h) / 2 - 2) c++; } return c; };
    for (const f of fs.readdirSync(ex).filter(f => f.endsWith('.json'))) {
      const g = JSON.parse(fs.readFileSync(ex + f, 'utf8')).graph; if (!g.nodes.some(n => n.type === 'block')) continue;
      const h = NN.explodeBlocks(g), n0 = cnt(g);
      const code = NN.generate(h).code, h2 = NN.importPython(code).graph;
      if (cnt(h) !== n0 || cnt(h2) !== n0) { console.error(`explode ${f}: ${cnt(h)} / ${cnt(h2)} parameters, expected ${n0}`); process.exit(1); }
      if (overlaps(h)) { console.error(`explode ${f}: ${overlaps(h)} layers overlap`); process.exit(1); }
      const keys = NN.viewKeys(h); if (new Set(keys.values()).size !== h.nodes.length) { console.error(`explode ${f}: view keys aren't unique`); process.exit(1); }
      // a block with more than three copies keeps the rest as a block; drawing it out (by name) leaves none of it
      for (const b of h.nodes.filter(n => n.type === 'block' && n.params.name)) {
        const all = NN.explodeBlocks(g, { open: { [b.params.name]: true } });
        if (all.nodes.some(n => n.type === 'block' && n.params.name === b.params.name) || cnt(all) !== n0) { console.error(`explode ${f}: drawing out ${b.params.name} didn't`); process.exit(1); }
        break;
      }
      // a saved view's positions come back on a fresh explosion, by code name
      const v = NN.viewPositions(h), k0 = Object.keys(v)[3]; v[k0] = [12345, -678];
      const h3 = NN.explodeBlocks(g); NN.applyViewPositions(h3, v);
      if (!h3.nodes.some(n => n.x === 12345 && n.y === -678)) { console.error(`explode ${f}: a saved position didn't come back`); process.exit(1); }
    }
  }
  console.log('explode and views OK');
}

// The page declares its encoding in its first bytes: opened as a file, a browser would otherwise guess (× became Ã—)
{
  const head = require('fs').readFileSync(__dirname + '/src.html').subarray(0, 1024).toString('latin1');
  if (!/<meta charset="utf-8">/i.test(head)) { console.error('src.html: no <meta charset="utf-8"> in its first 1024 bytes'); process.exit(1); }
  console.log('charset OK');
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
