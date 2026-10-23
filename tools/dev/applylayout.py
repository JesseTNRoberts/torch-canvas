# Applies the residual-stream layout (stream across the top, sublayer columns below) to examples.
# usage: python tools/dev/applylayout.py networks/23_residual_cnn_gap.json blocks/prenorm_conv_block.json
# Needs playwright and a built index.html.
import sys, json
from playwright.sync_api import sync_playwright
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); files = sys.argv[1:]
js = open(os.path.join(ROOT, 'tools', 'dev', 'reslayout.js')).read()
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium'); pg = b.new_page(viewport={'width':2100,'height':1150})
    errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('file://' + ROOT + '/index.html'); pg.wait_for_timeout(800); pg.add_script_tag(content=js)
    for f in files:
        res = pg.evaluate('''async (f) => { const {S, openLibrary, openLibraryBlock, sizeOf, render} = window.__tc; const e = S.library.find(e => e.file === f);
          if (e.kind==='network') await openLibrary(e); else await openLibraryBlock(e); render(); return window.__resLayout(S.g, sizeOf); }''', f)
        path = os.path.join(ROOT, 'examples', f); j = json.load(open(path)); g = j.get('graph', j)
        P = {r['id']: r for r in res}
        for n in g['nodes']:
            if n['id'] in P: n['x'], n['y'] = P[n['id']]['x'], P[n['id']]['y']
        for e in g['edges']: e.pop('route', None)
        json.dump(j, open(path, 'w'), indent=1, allow_nan=False); print(f, len(res), 'of', len(g['nodes']))
    print('errors', errs); b.close()
