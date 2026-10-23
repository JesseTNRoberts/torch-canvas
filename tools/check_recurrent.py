# The recurrent examples (14_rnn, 15_lstm, 16_gru), drawn flat with Delays, against nn.RNN / nn.LSTM / nn.GRU: the
# built-in layer's fused weights are split into the per-gate Linears (PyTorch's gate order), the readout is shared, and
# the outputs must match.
# needs: node, torch.  usage: python tools/check_recurrent.py
import importlib.util, os, subprocess, tempfile, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); TMP = tempfile.mkdtemp()
def generate(example):
    out = os.path.join(TMP, example.replace('.json', '.py'))
    js = "const NN=require(process.argv[1]),fs=require('fs');fs.writeFileSync(process.argv[3],NN.generate(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).graph).code)"
    subprocess.run(['node', '-e', js, os.path.join(ROOT, 'core.js'), os.path.join(ROOT, 'examples', 'networks', example), out], check=True)
    return out
torch.manual_seed(0)
def load(path, cls):
    spec = importlib.util.spec_from_file_location(cls, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return getattr(m, cls)()
def put(lin, w, b):
    with torch.no_grad(): lin.weight.copy_(w); lin.bias.copy_(b)
x = torch.randn(4, 20, 16)
for f, cls, ref, gates in [('14_rnn.py', 'RNN', nn.RNN(16, 64, batch_first=True), None), ('15_lstm.py', 'LSTM', nn.LSTM(16, 64, batch_first=True), 'ifgo'), ('16_gru.py', 'GRU', nn.GRU(16, 64, batch_first=True), 'rzn')]:
    m = load(generate(f.replace('.py', '.json')), cls)
    if gates is None:
        put(m.ih, ref.weight_ih_l0, ref.bias_ih_l0); put(m.hh, ref.weight_hh_l0, ref.bias_hh_l0)
    else:
        for k, g in enumerate(gates):
            sl = slice(64 * k, 64 * (k + 1))
            put(getattr(m, 'ih_' + g), ref.weight_ih_l0[sl], ref.bias_ih_l0[sl]); put(getattr(m, 'hh_' + g), ref.weight_hh_l0[sl], ref.bias_hh_l0[sl])
    with torch.no_grad(): a = m(x); b = m.out(ref(x)[0])
    np_ours, np_ref = sum(p.numel() for p in m.parameters()), sum(p.numel() for p in ref.parameters()) + sum(p.numel() for p in m.out.parameters())
    print(f"{cls}: {np_ours:,} parameters ({ref.__class__.__name__} + readout {np_ref:,}), max output difference {(a - b).abs().max().item():.2e}")
    assert np_ours == np_ref and (a - b).abs().max() < 1e-5
    # gradients flow through time too
    y = m(x.requires_grad_()); y.sum().backward(); assert x.grad.abs().sum() > 0; x.grad = None
print('all match')
