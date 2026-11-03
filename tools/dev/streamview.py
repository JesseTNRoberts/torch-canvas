# Saves an "Exploded" view into example networks: the model with its blocks drawn out (Explode blocks, default rule),
# laid out by the residual-stream convention (NN.streamLayout) with the app's real box sizes. Explode blocks then opens
# this view for the example. Tune a view by hand in the app instead (Views → Save as a view of …) when the stream
# layout isn't the clearest drawing; this script is for the residual networks it suits.
# usage (from the repo root, after npm run build): python tools/dev/streamview.py networks/07_gpt2.json …
import json, os, sys
from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
exe = '/opt/pw-browsers/chromium' if os.path.exists('/opt/pw-browsers/chromium') else None
with sync_playwright() as p:
    b = p.chromium.launch(**({'executable_path': exe} if exe else {})); pg = b.new_page()
    pg.goto('file://' + ROOT + '/index.html'); pg.wait_for_function('() => window.__tc && window.__tc.S.g')
    for f in sys.argv[1:]:
        res = pg.evaluate("""async (f) => { const t = window.__tc, e = t.S.library.find(x => x.file === f);
            await t.openLibrary(e); const src = t.S.root;
            const g = NN.explodeBlocks(src, { sz: t.sizeOf });
            if (!NN.streamLayout(g, t.sizeOf)) return null;
            return NN.viewPositions(g); }""", f)
        if not res:
            print(f, 'is not one residual stream: no view written'); continue
        path = os.path.join(ROOT, 'examples', f); j = json.load(open(path))
        j['graph'].setdefault('views', {})['Exploded'] = {'kind': 'exploded', 'open': {}, 'positions': res}
        json.dump(j, open(path, 'w'), indent=1, allow_nan=False); print(f, len(res), 'positions')
    b.close()
