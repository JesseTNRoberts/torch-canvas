# Runs the code Torch Canvas writes for every example network in real PyTorch, so a change in the app (or a new
# PyTorch) that breaks the generated code is caught.
#
#   node tools/export_examples.js out/ && python tools/check_examples.py out/ [--train] [--only NAME]
#
# For each example:
#   model.py and model_named.py: the code panel's code (sizes as numbers, and as constructor arguments). Its own
#     `__main__` block runs: build the model, count parameters, run a forward pass, print the output shapes. The
#     parameter count must be the app's, and every shape must be the one written in its "# expected" comment.
#     Big models (over SMALL parameters) run on the meta device, which checks shapes without memory or compute.
#   train.py, with --train (small models only): 2 optimizer steps on synthetic data with a checkpoint, then the same
#     script again resuming from that checkpoint; train_bf16.py: one step in bf16 autocast, SGD, one-cycle, EMA,
#     gradient accumulation and clipping.
# Exits non-zero if anything fails. Needs torch (CPU is enough).
import contextlib, gc, io, json, os, re, shutil, subprocess, sys, tempfile, time, traceback
import torch

SMALL = 40_000_000  # bigger models (or activations) would need too much memory on a CI runner
out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
if not out:
    sys.exit('usage: python tools/check_examples.py out/ [--train] [--only NAME]')
train = '--train' in sys.argv
only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
manifest = json.load(open(os.path.join(out, 'manifest.json')))


def run_main(path, meta):
    """Runs a model file's __main__ block; returns (stdout lines, expected shapes from its comments)."""
    src = open(path).read()
    main = src[src.index('if __name__ == "__main__":'):]
    expected = re.findall(r'# expected (torch\.Size\(\[[^\]]*\]\))', main)
    if meta:  # a meta tensor has no value: print it instead of reading it
        src = src.replace('.item()', '')
    buf = io.StringIO()
    g = {'__name__': '__main__', '__file__': path}
    with contextlib.redirect_stdout(buf):
        if meta:
            with torch.device('meta'):
                exec(compile(src, path, 'exec'), g)
        else:
            torch.manual_seed(0)
            with torch.no_grad():
                exec(compile(src, path, 'exec'), g)
    g.clear(); gc.collect()
    return buf.getvalue().splitlines(), expected


fails, t0 = [], time.time()
for e in manifest:
    name = e['name']
    if only and only not in name:
        continue
    d = os.path.join(out, name)
    meta = e['params'] > SMALL
    for f in ('model.py', 'model_named.py'):
        try:
            lines, expected = run_main(os.path.join(d, f), meta)
            got_params = int(lines[0].split()[0])
            if got_params != e['params']:
                raise AssertionError(f'{got_params} parameters in PyTorch, the app shows {e["params"]}')
            shapes = [l.strip() for l in lines if l.strip().startswith('torch.Size')]
            if shapes != expected:
                raise AssertionError(f'output shapes {shapes}, the code says {expected}')
            print(f'ok   {name}/{f}{" (meta)" if meta else ""}')
        except NotImplementedError as ex:
            # a forward pass that reads values (Qwen2.5-VL's image grid) can't run on meta: check the build alone
            if not meta or 'meta' not in str(ex):
                fails.append(f'{name}/{f}: {ex}'); print(f'FAIL {name}/{f}: {ex}'); continue
            try:
                src = open(os.path.join(d, f)).read()
                cls = re.search(r'model = (\w+)\(\)', src).group(1)
                g = {'__name__': 'checked'}
                exec(compile(src, f, 'exec'), g)
                with torch.device('meta'):
                    n = sum(p.numel() for p in g[cls]().parameters())
                if n != e['params']:
                    raise AssertionError(f'{n} parameters in PyTorch, the app shows {e["params"]}')
                print(f'ok   {name}/{f} (meta, built only: its forward pass reads values)')
            except Exception as ex2:
                fails.append(f'{name}/{f}: {ex2}'); print(f'FAIL {name}/{f}: {ex2}')
        except Exception as ex:
            fails.append(f'{name}/{f}: {ex}')
            print(f'FAIL {name}/{f}: {ex}')
            traceback.print_exc(limit=3)
    if train and e['train'] and not meta:
        wd = tempfile.mkdtemp()
        open(os.path.join(wd, 'model.py'), 'w').write(open(os.path.join(d, 'model.py')).read())
        for script, resume in (('train.py', False), ('train.py', True), ('train_bf16.py', False)):
            src = open(os.path.join(d, script)).read()
            if resume:
                src = src.replace('RESUME = None', 'RESUME = "ckpt/last.pt"', 1)
            open(os.path.join(wd, 'run.py'), 'w').write(src)
            tag = f'{name}/{script}{" (resume)" if resume else ""}'
            r = subprocess.run([sys.executable, 'run.py'], cwd=wd, capture_output=True, text=True, timeout=900)
            if r.returncode != 0 or (resume and 'resumed from' not in r.stdout):
                fails.append(f'{tag}: ' + (r.stderr.strip().splitlines() or ['no "resumed from" in its output'])[-1])
                print(f'FAIL {tag}\n{r.stderr[-1500:]}')
            else:
                print(f'ok   {tag}')
        shutil.rmtree(wd, ignore_errors=True)

print(f'\n{len(fails)} failure(s) in {time.time() - t0:.0f}s')
for f in fails:
    print(' -', f)
sys.exit(1 if fails else 0)
