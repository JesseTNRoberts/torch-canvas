# Measures what the variance plot predicts, by running the model: the variance of the input and output of every
# Linear, Conv2d, ConvTranspose2d and Conv1d (forward, standard Gaussian input) and of the gradients there (backward, a standard Gaussian
# gradient at every output), averaged over several random initializations. BatchNorm is in training mode (batch
# statistics) and dropout is off, as in the plot.
# usage: python tools/measure_variance.py model.py ClassName "8, 3, 224, 224" [seeds] [--ids VOCAB]
#   --positions N: an input of rank 2 is positions (distinct, in [0, N)), like I-JEPA's masks
#   several inputs: their shapes separated by ";" ("8, 64; 8, 48" for a translation model's source and target)
#   --ids VOCAB: the input is token ids in [0, VOCAB) instead of Gaussian values
# prints JSON: [{name, fwdIn, fwdOut, bwdIn, bwdOut}, ...] in the order the layers run
import gc, importlib.util, json, sys, torch, torch.nn as nn

path, cls = sys.argv[1], sys.argv[2]
shapes = [[int(v) for v in part.split(',')] for part in sys.argv[3].split(';')]
seeds = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 4
spec = importlib.util.spec_from_file_location('m', path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
acc = None
for seed in range(seeds):
    torch.manual_seed(seed)
    model = getattr(mod, cls)(); model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.eval()
    names = {m: n for n, m in model.named_modules()}
    rec, order = {}, []
    def fwd(m, i, o):
        if m not in rec: rec[m] = {}; order.append(m)
        rec[m]['fwdIn'] = i[0].detach().var(unbiased=False).item(); rec[m]['fwdOut'] = o.detach().var(unbiased=False).item()
    # each layer's own share of the gradient (a tensor read by several layers gets the sum of theirs)
    # a layer that runs at every step of a loop: its forward values come from the last step, and so do its gradients
    # (backward runs the last step first, so the first call is kept)
    def bwd(m, gin, gout):
        if 'bwdOut' in rec[m]: return
        rec[m]['bwdOut'] = gout[0].var(unbiased=False).item()
        if gin[0] is not None: rec[m]['bwdIn'] = gin[0].var(unbiased=False).item()
    layers = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d))]
    hs = [m.register_forward_hook(fwd) for m in layers] + [m.register_full_backward_hook(bwd) for m in layers]
    ids = sys.argv.index('--ids') if '--ids' in sys.argv else None
    pos = int(sys.argv[sys.argv.index('--positions') + 1]) if '--positions' in sys.argv else None
    xs = [torch.rand(sh[0], pos).argsort(dim=1)[:, :sh[1]] if pos and len(sh) == 2 else torch.randint(0, int(sys.argv[ids + 1]), sh) if ids else torch.randn(*sh, requires_grad=True) for sh in shapes]
    outs = model(*xs); outs = outs if isinstance(outs, (tuple, list)) else [outs]
    sum((o * torch.randn_like(o)).sum() for o in outs).backward()
    rows = [dict(name=names[m], **rec[m]) for m in order]
    if acc is None: acc = rows
    else:
        for a, r in zip(acc, rows):
            for k in ('fwdIn', 'fwdOut', 'bwdIn', 'bwdOut'): a[k] = a.get(k, 0) + r.get(k, 0)
    for h in hs: h.remove()
    del model, outs, xs, rec, order, names; gc.collect()  # a big model's activations and weights before the next seed
for a in acc:
    for k in ('fwdIn', 'fwdOut', 'bwdIn', 'bwdOut'):
        if k in a: a[k] /= seeds
print(json.dumps(acc))
