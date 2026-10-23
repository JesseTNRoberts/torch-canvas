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
for (const [f, want] of [['23_residual_cnn_gap.json', 57098], ['24_residual_cnn_dense.json', 72458]]) {
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
