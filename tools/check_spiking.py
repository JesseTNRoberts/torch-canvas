# The spiking examples, drawn flat with Delays, against plain loops written here and against snnTorch.
#   26_lif            snnTorch's Leaky: the same weights and input must give the same spikes, and Spike's surrogate
#                     gradient must be snnTorch's fast sigmoid
#   27_lif_axonal     the LIF's spikes, 3 steps late; a gradient at step t lands where the plain LIF's does at t - 3
#   28_lif_synaptic   snnTorch's Synaptic
#   29_lif_recurrent  snnTorch's RLeaky (all-to-all, its recurrent weights copied in, no bias)
#   30_lif_adaptive   a plain loop (snnTorch has no neuron of this form)
# Every example is also checked against a plain loop written here, values and gradients, so the check still says
# something without snnTorch installed.
# needs: node, torch; snntorch for the comparisons with it.  usage: python tools/check_spiking.py
import importlib.util, os, subprocess, tempfile, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); TMP = tempfile.mkdtemp()
BETA, THETA, SLOPE = 0.9, 1.0, 25.0
js = "const NN=require(process.argv[1]),fs=require('fs');fs.writeFileSync(process.argv[3],NN.generate(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).graph).code)"
def load(example):   # the code the app writes for an example, as a module
    out = os.path.join(TMP, example + '.py')
    subprocess.run(['node', '-e', js, os.path.join(ROOT, 'core.js'), os.path.join(ROOT, 'examples', 'networks', example + '.json'), out], check=True)
    spec = importlib.util.spec_from_file_location(example, out); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
mod = load('26_lif')
torch.manual_seed(0)
m = mod.LIF()
x = 3 * torch.randn(4, 20, 16)   # large enough that neurons fire: about a tenth of the outputs are spikes

# 1. Spike: exactly 0 or 1 forward; backward 1 / (1 + slope |x|)^2
z = torch.linspace(-2, 2, 401, requires_grad=True)
s = mod.spike(z); s.sum().backward()
want = 1 / (1 + SLOPE * z.detach().abs()) ** 2
assert set(s.detach().unique().tolist()) <= {0.0, 1.0} and torch.equal(s.detach(), (z.detach() > 0).float())
print(f"spike: values are exactly 0 or 1; gradient vs 1/(1 + {SLOPE:g}|x|)^2, max difference {(z.grad - want).abs().max().item():.2e}")
assert (z.grad - want).abs().max() < 1e-6

# 2. the example against a plain loop: spikes, and gradients to the weights and the input through time
class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v): ctx.save_for_backward(v); return (v > 0).to(v.dtype)
    @staticmethod
    def backward(ctx, g): (v,) = ctx.saved_tensors; return g / (1 + SLOPE * v.abs()) ** 2
def plain(x, w):
    cur = x @ w.t(); v = torch.zeros(x.size(0), w.size(0)); s = torch.zeros_like(v); ss = []
    for t in range(x.size(1)):
        v = cur[:, t] + BETA * v - THETA * s
        s = Step.apply(v - THETA); ss.append(s)
    return torch.stack(ss, dim=1)
readout = torch.randn(64)
xa = x.clone().requires_grad_(); a = m(xa); (a * readout).sum().backward(); ga_w, ga_x = m.ih.weight.grad.clone(), xa.grad.clone()
w = m.ih.weight.detach().clone().requires_grad_(); xb = x.clone().requires_grad_(); b = plain(xb, w); (b * readout).sum().backward()
print(f"LIF vs a plain loop: {int(a.sum())} spikes of {a.numel()} ({a.mean().item():.1%}), {int((a != b).sum())} differ; "
      f"gradient difference, weights {(ga_w - w.grad).abs().max().item():.2e}, input {(ga_x - xb.grad).abs().max().item():.2e}")
assert torch.equal(a.detach(), b.detach()) and a.sum() > 100 and ga_w.abs().sum() > 0
assert (ga_w - w.grad).abs().max() < 1e-4 * ga_w.abs().max() and (ga_x - xb.grad).abs().max() < 1e-4 * ga_x.abs().max()
assert sum(p.numel() for p in m.parameters()) == 1024

# 3. the other spiking examples against plain loops: spikes exactly, gradients closely
ALPHA, RHO, B, AXON = 0.8, 0.95, 0.2, 3
def plain_synaptic(x, w):
    cur = x @ w.t(); i = torch.zeros(x.size(0), w.size(0)); v = torch.zeros_like(i); s = torch.zeros_like(i); ss = []
    for t in range(x.size(1)):
        i = cur[:, t] + ALPHA * i
        v = i + BETA * v - THETA * s
        s = Step.apply(v - THETA); ss.append(s)
    return torch.stack(ss, dim=1)
def plain_recurrent(x, w, r):
    cur = x @ w.t(); v = torch.zeros(x.size(0), w.size(0)); s = torch.zeros_like(v); ss = []
    for t in range(x.size(1)):
        v = cur[:, t] + s @ r.t() + BETA * v - THETA * s
        s = Step.apply(v - THETA); ss.append(s)
    return torch.stack(ss, dim=1)
def plain_adaptive(x, w):
    cur = x @ w.t(); v = torch.zeros(x.size(0), w.size(0)); s = torch.zeros_like(v); a = torch.zeros_like(v); ss = []
    for t in range(x.size(1)):
        a = RHO * a + s
        v = cur[:, t] + BETA * v - THETA * s
        s = Step.apply(v - THETA - B * a); ss.append(s)
    return torch.stack(ss, dim=1)
def against_plain(name, model, ref, extra=()):
    xa = x.clone().requires_grad_(); ya = model(xa); (ya * readout).sum().backward()
    ws = [p.detach().clone().requires_grad_() for p in [model.ih.weight, *extra]]; xb = x.clone().requires_grad_(); yb = ref(xb, *ws); (yb * readout).sum().backward()
    gw = max((p.grad - q.grad).abs().max().item() / p.grad.abs().max().item() for p, q in zip([model.ih.weight, *extra], ws))
    gx = ((xa.grad - xb.grad).abs().max() / xa.grad.abs().max()).item()
    print(f"{name} vs a plain loop: {int(ya.sum())} spikes of {ya.numel()} ({ya.mean().item():.1%}), {int((ya != yb).sum())} differ; relative gradient difference, weights {gw:.2e}, input {gx:.2e}")
    assert torch.equal(ya.detach(), yb.detach()) and ya.sum() > 100 and gw < 1e-4 and gx < 1e-4
    return ya.detach()
syn = load('28_lif_synaptic').SynapticLIF(); y_syn = against_plain('synaptic LIF', syn, plain_synaptic)
rec = load('29_lif_recurrent').RecurrentLIF(); y_rec = against_plain('recurrent LIF', rec, plain_recurrent, [rec.rec.weight])
ada = load('30_lif_adaptive').AdaptiveLIF(); y_ada = against_plain('adaptive LIF', ada, plain_adaptive)
assert sum(p.numel() for p in syn.parameters()) == 1024 and sum(p.numel() for p in rec.parameters()) == 5120 and sum(p.numel() for p in ada.parameters()) == 1024
# the adaptive threshold does something: fewer spikes than the same weights without it
with torch.no_grad(): fewer = int(plain(x, ada.ih.weight).sum()) - int(y_ada.sum())
print(f"adaptive LIF: {fewer} fewer spikes than the same layer with a fixed threshold"); assert fewer > 0

# 4. the axon: the LIF's spikes a whole number of steps late, and gradients that land at the right step
axmod = load('27_lif_axonal'); ax = axmod.AxonalLIF(); ax.load_state_dict(m.state_dict())
with torch.no_grad(): late = ax(x)
print(f"axonal LIF: output is the LIF's, {AXON} steps late: {bool(torch.equal(late[:, AXON:], a.detach()[:, :-AXON]))}; first {AXON} steps silent: {bool(late[:, :AXON].abs().sum() == 0)}")
assert torch.equal(late[:, AXON:], a.detach()[:, :-AXON]) and late[:, :AXON].abs().sum() == 0 and sum(p.numel() for p in ax.parameters()) == 1024
T0 = 12   # a loss on one step of the delayed output is a loss on the LIF's output AXON steps earlier
x1 = x.clone().requires_grad_(); (ax(x1)[:, T0] * readout).sum().backward()
x2 = x.clone().requires_grad_(); (m(x2)[:, T0 - AXON] * readout).sum().backward()
last = max(t for t in range(x.size(1)) if x1.grad[:, t].abs().sum() > 0)
print(f"axon gradient: a loss at step {T0} of the delayed output gives the input the gradient of a loss at step {T0 - AXON} of the LIF (max difference {(x1.grad - x2.grad).abs().max().item():.2e}); the last input step that gets any is {last}")
assert (x1.grad - x2.grad).abs().max() < 1e-6 and x1.grad.abs().sum() > 0 and last == T0 - AXON
d = axmod.delay(torch.arange(24.).reshape(1, 6, 4), 2)   # the helper itself: a shift by whole steps, zeros in front
assert torch.equal(d[:, 2:], torch.arange(24.).reshape(1, 6, 4)[:, :-2]) and d[:, :2].abs().sum() == 0 and axmod.delay(torch.ones(1, 3, 2), 5).abs().sum() == 0

# 5. against snnTorch
try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    print('snntorch is not installed: skipped the comparison with snn.Leaky'); raise SystemExit(0)
zs = torch.linspace(-2, 2, 401, requires_grad=True)
surrogate.fast_sigmoid(slope=SLOPE)(zs).sum().backward()
print(f"spike gradient vs snntorch.surrogate.fast_sigmoid(slope={SLOPE:g}): max difference {(z.grad - zs.grad).abs().max().item():.2e}")
assert (z.grad - zs.grad).abs().max() < 1e-6
lif = snn.Leaky(beta=BETA, threshold=THETA, spike_grad=surrogate.fast_sigmoid(slope=SLOPE))   # reset by subtraction
xs = x.clone().requires_grad_(); cur = m.ih(xs); mem = torch.zeros(4, 64); ref, mems = [], []
for t in range(x.size(1)):
    spk, mem = lif(cur[:, t], mem); ref.append(spk); mems.append(mem)
ref = torch.stack(ref, dim=1)
print(f"LIF vs snn.Leaky(beta={BETA}, threshold={THETA:g}): {int((a.detach() != ref.detach()).sum())} of {ref.numel()} outputs differ")
assert torch.equal(a.detach(), ref.detach())
# snnTorch detaches the reset (no gradient through - threshold * last spike); the drawing keeps it, so gradients differ
m.ih.weight.grad = None; (ref * readout).sum().backward()
rel = lambda p, q: ((p - q).norm() / q.norm()).item()
print(f"gradients vs snn.Leaky, which detaches the reset: relative difference, weights {rel(ga_w, m.ih.weight.grad):.2f}, input {rel(ga_x, xs.grad):.2f}")
# Synaptic: syn = alpha * syn + input, mem = beta * mem + syn - reset
sg = surrogate.fast_sigmoid(slope=SLOPE)
lif2 = snn.Synaptic(alpha=ALPHA, beta=BETA, threshold=THETA, spike_grad=sg)
with torch.no_grad():
    cur = syn.ih(x); sy = torch.zeros(4, 64); mem = torch.zeros(4, 64); ref = []
    for t in range(x.size(1)):
        spk, sy, mem = lif2(cur[:, t], sy, mem); ref.append(spk)
    ref = torch.stack(ref, dim=1)
print(f"synaptic LIF vs snn.Synaptic(alpha={ALPHA}, beta={BETA}): {int((y_syn != ref).sum())} of {ref.numel()} outputs differ")
assert torch.equal(y_syn, ref)
# RLeaky, all to all: mem = beta * mem + input + V(last spike) - reset, with the example's recurrent weights and no bias
lif3 = snn.RLeaky(beta=BETA, linear_features=64, threshold=THETA, spike_grad=sg, all_to_all=True)
with torch.no_grad():
    lif3.recurrent.weight.copy_(rec.rec.weight)
    if lif3.recurrent.bias is not None: lif3.recurrent.bias.zero_()
    cur = rec.ih(x); spk = torch.zeros(4, 64); mem = torch.zeros(4, 64); ref = []
    for t in range(x.size(1)):
        spk, mem = lif3(cur[:, t], spk, mem); ref.append(spk)
    ref = torch.stack(ref, dim=1)
print(f"recurrent LIF vs snn.RLeaky(beta={BETA}, all_to_all=True): {int((y_rec != ref).sum())} of {ref.numel()} outputs differ")
assert torch.equal(y_rec, ref)
print('all match')
