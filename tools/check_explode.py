# Checks that Explode blocks gives the same network: the exploded model, with the original's weights copied in,
# gives the same outputs. Weight names are matched across the two (block1.conv1.weight → block1_conv1.weight; in a
# ModuleList whose first copy was drawn out, h.0.attn → h_0_attn and h.3 → h.2).
#
#   node tools/export_examples.js out/ && python tools/check_explode.py out/ [--max-params N]
import contextlib, io, itertools, json, os, re, sys
import torch

out = sys.argv[1] if len(sys.argv) > 1 else sys.exit('usage: python tools/check_explode.py out/ [--max-params N]')
MAX = int(sys.argv[sys.argv.index('--max-params') + 1]) if '--max-params' in sys.argv else 130_000_000
fails = []


def load(path):
    src = open(path).read()
    g = {'__name__': 'checked'}
    exec(compile(src, path, 'exec'), g)
    cls = re.search(r'model = (\w+)\(\)', src).group(1)
    main = src[src.index('if __name__ == "__main__":'):]
    return g, cls, main


def inputs(g, cls, main):
    """The example inputs the file's own __main__ block makes (it is run with a stand-in model that records them)."""
    class Cap:
        args = None
        def parameters(self): return iter([])
        def __call__(self, *a): Cap.args = a; raise StopIteration
    ns = dict(g); ns[cls] = lambda: Cap()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(main.replace('if __name__ == "__main__":', 'if True:'), ns)
        except StopIteration:
            pass
    return Cap.args


def joins(seg):
    """seg joined every way with . or _ (a drawn-out block's path is joined with _)."""
    for js in itertools.product('._', repeat=max(0, len(seg) - 1)):
        yield seg[0] + ''.join(j + x for j, x in zip(js, seg[1:]))


def candidates(key, names):
    """Names the original's weight can have in the exploded model, by the code writer's rules: a drawn-out block's
    path is joined with _ (block1.conv1 → block1_conv1); a list L whose first copy was drawn out has L_0_… for its old
    L.0 and L.0 … for its old L.1 … (just L … when one copy is left: one copy isn't written as a list)."""
    seg = key.split('.')
    for i, s in enumerate(seg):
        if i == 0 or not s.isdigit():
            continue
        heads = list(joins(seg[:i]))
        if any(n.startswith(h + '_0_') or n.startswith(h + '_0.') for h in heads for n in names) and any(n.startswith(h + '.') and not n.startswith(h + '_') for h in heads for n in names):
            k = int(s)
            if k == 0:
                yield from joins(seg[:i - 1] + [seg[i - 1] + '_0'] + seg[i + 1:])
            else:
                yield from joins(seg[:i] + [str(k - 1)] + seg[i + 1:])
                yield from joins(seg[:i] + seg[i + 1:])
            return
    yield from joins(seg)


man = json.load(open(os.path.join(out, 'manifest.json')))
for e in man:
    d = os.path.join(out, e['name'])
    if not os.path.exists(os.path.join(d, 'model_exploded.py')) or e['params'] > MAX:
        continue
    try:
        ga, ca, ma = load(os.path.join(d, 'model.py'))
        gb, cb, mb = load(os.path.join(d, 'model_exploded.py'))
        torch.manual_seed(0)
        A = ga[ca](); B = gb[cb]()
        sa, sb = A.state_dict(), B.state_dict()
        mapped, taken = {}, set()
        for k, v in sa.items():
            hit = next((c for c in candidates(k, sb) if c in sb and c not in taken and sb[c].shape == v.shape), None)
            if hit is None:
                raise AssertionError(f'no place for {k} in the exploded model')
            mapped[hit] = v; taken.add(hit)
        if len(taken) != len(sb):
            raise AssertionError(f'{len(sb) - len(taken)} weights of the exploded model have no counterpart')
        B.load_state_dict(mapped)
        A.eval(); B.eval()
        torch.manual_seed(1)
        x = inputs(ga, ca, ma)
        with torch.no_grad():
            ya, yb = A(*x), B(*x)
        ya = ya if isinstance(ya, (tuple, list)) else (ya,)
        yb = yb if isinstance(yb, (tuple, list)) else (yb,)
        diff = max(float((p - q).abs().max()) for p, q in zip(ya, yb))
        if diff > 1e-5 * max(1.0, max(float(p.abs().max()) for p in ya)):
            raise AssertionError(f'outputs differ by {diff:.3g}')
        print(f"ok   {e['name']}: same outputs ({len(sb)} weights matched)", flush=True)
    except Exception as ex:
        fails.append(f"{e['name']}: {ex}")
        print(f"FAIL {e['name']}: {ex}", flush=True)

print(f'\n{len(fails)} failure(s)')
for f in fails:
    print(' -', f)
    if os.environ.get('GITHUB_ACTIONS'):
        print('::error title=explode check::' + f[:900])
sys.exit(1 if fails else 0)
