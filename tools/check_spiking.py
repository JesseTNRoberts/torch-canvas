# The spiking example (26_lif), drawn flat with two Delays, against snnTorch's Leaky neuron: the same weights and input
# must give the same spikes, and Spike's surrogate gradient must be snnTorch's fast sigmoid. Also checked against a
# plain loop written here, values and gradients, so the check still says something without snnTorch installed.
# needs: node, torch; snntorch for the comparison with Leaky.  usage: python tools/check_spiking.py
import importlib.util, os, subprocess, tempfile, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); TMP = tempfile.mkdtemp()
BETA, THETA, SLOPE = 0.9, 1.0, 25.0
out = os.path.join(TMP, 'lif.py')
js = "const NN=require(process.argv[1]),fs=require('fs');fs.writeFileSync(process.argv[3],NN.generate(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).graph).code)"
subprocess.run(['node', '-e', js, os.path.join(ROOT, 'core.js'), os.path.join(ROOT, 'examples', 'networks', '26_lif.json'), out], check=True)
spec = importlib.util.spec_from_file_location('lif', out); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
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

# 3. against snnTorch
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
print('all match')
