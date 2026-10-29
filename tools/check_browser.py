# Opens the built app (index.html) in headless Chromium and checks what a person would hit first:
#   - it starts with no script errors, and opens every example network and library block, draws it, writes its code,
#     and plots its variance without errors
#   - it still starts when what it saved in the browser is damaged (bad JSON, wrong shapes, half-written graphs)
#   - text from pasted code (strings, comments, a category line) is shown as text, never run as HTML
#   - a full browser storage doesn't stop editing, and the person is told their tabs aren't being kept
#
#   npm run build && python tools/check_browser.py [--quick]
# Needs playwright (pip install playwright; python -m playwright install chromium, or set CHROMIUM to a browser).
import os, sys, pathlib
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = (ROOT / 'index.html').as_uri()
QUICK = '--quick' in sys.argv
# examples whose variance plot is known not to work yet, and the message it gives instead
VARIANCE_UNSUPPORTED = {'networks/10_qwen2_5_vl_3b.json': "Can't predict through vision_window_index"}
fails = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what)
    if not cond:
        fails.append(what)


def new_page(browser, storage=None):
    ctx = browser.new_context(viewport={'width': 1600, 'height': 1000})
    if storage is not None:  # what an earlier visit left in localStorage
        ctx.add_init_script('(() => { if (sessionStorage.getItem("seeded")) return; sessionStorage.setItem("seeded", "1"); localStorage.clear(); '
                            + ''.join(f'localStorage.setItem({k!r}, {v!r});' for k, v in storage.items()) + '})()')
    pg = ctx.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.on('dialog', lambda d: (errs.append('dialog: ' + d.message), d.dismiss()))
    pg.goto(URL)
    pg.wait_for_function('() => window.__tc && window.__tc.S && window.__tc.S.g', timeout=15000)
    return ctx, pg, errs


exe = os.environ.get('CHROMIUM') or ('/opt/pw-browsers/chromium' if os.path.exists('/opt/pw-browsers/chromium') else None)
with sync_playwright() as p:
    browser = p.chromium.launch(**({'executable_path': exe} if exe else {}))

    # ── every example ──
    ctx, pg, errs = new_page(browser, {})
    check(not errs, 'starts without errors' + (f': {errs}' if errs else ''))
    entries = pg.evaluate("() => window.__tc.S.library.map(e => ({ file: e.file, kind: e.kind, title: e.title }))")
    check(len(entries) > 40, f'{len(entries)} examples and blocks in the library')
    for e in entries:
        before = len(errs)
        res = pg.evaluate("""async (f) => { const t = window.__tc, e = t.S.library.find(x => x.file === f);
            if (e.kind === 'network') await t.openLibrary(e); else await t.openLibraryBlock(e);
            t.render(); t.fit();
            const code = document.querySelector('#code').textContent;
            return { nodes: t.S.g.nodes.length, drawn: document.querySelectorAll('svg [data-node]').length, code: code.length,
                     couldNot: /Could not generate code/.test(code) }; }""", e['file'])
        check(res['nodes'] > 0 and res['drawn'] > 0 and res['code'] > 100 and not res['couldNot'] and len(errs) == before,
              f"{e['file']}: {res['nodes']} nodes drawn, code written" + (f' {errs[before:]}' if len(errs) > before else ''))
        if e['kind'] == 'network' and not QUICK:
            before = len(errs)
            pg.click('#zVar')
            ok = True
            try:
                pg.wait_for_function("() => { const p = document.querySelector('#varPane'); return p && p.querySelector('svg.var-plot, .var-err') && !p.classList.contains('var-busy'); }", timeout=90000)
            except Exception:
                ok = False
            err_shown = pg.evaluate("() => { const e = document.querySelector('#varPane .var-err'); return e ? e.textContent : ''; }")
            pg.click('#varClose')
            known = VARIANCE_UNSUPPORTED.get(e['file'])
            if known and known in err_shown:
                err_shown = ''  # the message it is known to give (a limitation, not a crash)
            elif known:
                err_shown = err_shown or f'no longer "{known}…": update VARIANCE_UNSUPPORTED'
            check(ok and len(errs) == before and not err_shown, f"{e['file']}: variance plot" + (f' {errs[before:]}' if len(errs) > before else '') + (f' says "{err_shown}"' if err_shown else ''))
    ctx.close()

    # ── damaged saved state ──
    STORE, LIB = 'torchcanvas.graph.v1', 'torchcanvas.libblocks.v1'
    damaged = {
        'not JSON': {STORE: '{"v":2,"docs":[{'},
        'docs not a list': {STORE: '{"v":2,"docs":5,"from":3}'},
        'node without params': {STORE: '{"v":2,"docs":[{"name":"M","symbols":{},"nodes":[{"id":"a","type":"linear"}],"edges":[]}],"from":[null]}'},
        'edge to a missing node': {STORE: '{"v":2,"docs":[{"name":"M","symbols":{"e":8},"nodes":[{"id":"a","type":"input","x":0,"y":0,"params":{"name":"x","shape":"e, 4"}}],"edges":[{"id":"q","from":"a","to":"zz","port":0}],"groups":[]}],"from":[null]}'},
        'unknown node type': {STORE: '{"v":2,"docs":[{"name":"M","symbols":{},"nodes":[{"id":"a","type":"warpdrive","x":0,"y":0,"params":{}}],"edges":[],"groups":[]}],"from":[null]}'},
        'block use of a missing block': {STORE: '{"v":2,"docs":[{"name":"M","symbols":{"e":8},"nodes":[{"id":"a","type":"block","x":0,"y":0,"params":{"block":"nope","args":{}}}],"edges":[],"groups":[],"blocks":{}}],"from":[null]}'},
        'positions not numbers': {STORE: '{"v":2,"docs":[{"name":"M","symbols":{"e":8},"nodes":[{"id":"a","type":"input","x":"left","y":null,"params":{"name":"x","shape":"e, 4"}}],"edges":[],"groups":[]}],"from":[null]}'},
        'library overrides damaged': {LIB: '{"blocks/swiglu.py":{"name":5,"graph":{"nodes":"x"}}}'},
        'cur out of range': {STORE: '{"v":2,"docs":[],"from":[],"cur":99}'},
    }
    for what, storage in damaged.items():
        ctx, pg, errs = None, None, []
        try:
            ctx, pg, errs = new_page(browser, storage)
            started = pg.evaluate("() => window.__tc.S.g.nodes.length >= 0 && !!document.querySelector('svg')")
        except Exception as ex:
            started = False
            errs.append(str(ex).splitlines()[0])
        check(started and not errs, f'starts with damaged saved data ({what})' + (f': {errs[:2]}' if errs else ''))
        if ctx:
            ctx.close()

    # ── tabs come back after a reload: an untouched example (kept as a reference) and an edited model ──
    ctx, pg, errs = new_page(browser, {})
    pg.evaluate("() => { const t = window.__tc; t.S.sel.nodes.clear(); t.S.sel.nodes.add(t.S.g.nodes[1].id); t.selectionChanged(); }")
    pg.keyboard.press('Delete'); pg.wait_for_timeout(200)
    edited = pg.evaluate("() => [window.__tc.S.root.name, window.__tc.S.g.nodes.length]")
    pg.evaluate("async () => { const t = window.__tc; await t.openLibrary(t.S.library.find(e => e.file === 'networks/07_gpt2.json')); }")
    pg.wait_for_timeout(300)
    ref = pg.evaluate("() => JSON.parse(localStorage.getItem('torchcanvas.graph.v1')).docs.map(d => d.libRef || d.name)")
    pg.reload(); pg.wait_for_function('() => window.__tc && window.__tc.S.g')
    docs = pg.evaluate("() => window.__tc.S.docs.map(d => { const r = d.id === window.__tc.S.cur ? window.__tc.S.root : d.root; return [r.name, r.nodes.length]; })")
    check(docs == [edited, docs[1]] and docs[1][0] == 'GPT2' and 'networks/07_gpt2.json' in ref and not errs,
          f'tabs come back after a reload (saved as {ref}, back as {docs})' + (f' {errs}' if errs else ''))
    ctx.close()

    # ── a start that stopped part-way last time: the saved state is set aside and the app starts fresh ──
    old = str(int(__import__('time').time() * 1000) - 60000)
    ctx, pg, errs = new_page(browser, {'torchcanvas.booting': old, STORE: '{"v":2,"docs":[{"name":"Saved","symbols":{},"nodes":[],"edges":[]}],"from":[null]}'})
    pg.wait_for_timeout(700)
    msg = pg.evaluate("() => document.querySelector('#toast').textContent")
    keys = pg.evaluate("() => Object.keys(localStorage)")
    check('started fresh' in msg and any(k.startswith('torchcanvas.backup.') for k in keys) and 'torchcanvas.booting' not in keys and not errs,
          f'recovers from a start that stopped part-way (toast: "{msg[:80]}")')
    ctx.close()

    # ── text from code is text ──
    ctx, pg, errs = new_page(browser, {})
    evil = '<img src=x onerror=window.__pwned=1>'
    code = f'''# category: {evil}
import torch
import torch.nn as nn
import torch.nn.functional as F


class Evil(nn.Module):
    """{evil}"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        y = torch.myop(x, "{evil}")  # {evil}
        y = self.fc(y)
        return torch.cat([y, y], dim=-1)
'''
    pg.evaluate("(t) => { const dt = new DataTransfer(); dt.setData('text/plain', t); document.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true })); }", code)
    pg.wait_for_timeout(500)
    ids = pg.evaluate("() => window.__tc.S.g.nodes.map(n => n.id)")
    for i in ids:  # each node's details panel
        pg.evaluate("(id) => { const t = window.__tc; t.S.sel.nodes.clear(); t.S.sel.nodes.add(id); t.S.drawerOpen = true; t.selectionChanged(); }", i)
    pg.click('#bMore'); pg.wait_for_timeout(200); pg.keyboard.press('Escape')
    pwned = pg.evaluate("() => !!window.__pwned || !!document.querySelector('img[src=x]')")
    name = pg.evaluate("() => window.__tc.S.root.name")
    check(name == 'Evil' and len(ids) > 2 and not pwned and not errs, f'text in pasted code stays text ({name}, {len(ids)} nodes)' + (f' {errs}' if errs else ''))
    ctx.close()

    # ── full storage ──
    ctx, pg, errs = new_page(browser, {})
    pg.evaluate("""() => { let i = 0; for (let size = 1 << 20; size >= 1; size >>= 1) { const s = 'x'.repeat(size); try { for (;;) localStorage.setItem('filler' + i++, s); } catch (e) {} } }""")
    # an example tab is kept as a reference until it is edited; editing a big one needs room the storage hasn't got
    pg.evaluate("async () => { const t = window.__tc; await t.openLibrary(t.S.library.find(e => e.file === 'networks/21_transformer_base.json')); }")
    pg.wait_for_timeout(300)
    n0 = pg.evaluate("() => window.__tc.S.g.nodes.length")
    pg.evaluate("() => { const t = window.__tc; t.S.sel.nodes.clear(); t.S.sel.nodes.add(t.S.g.nodes[1].id); t.selectionChanged(); }")
    pg.keyboard.press('Delete'); pg.wait_for_timeout(300)
    n1 = pg.evaluate("() => window.__tc.S.g.nodes.length")
    toast = pg.evaluate("() => [...document.querySelectorAll('.toast, #toast')].map(t => t.textContent).join(' ')")
    check(n1 == n0 - 1 and not errs, 'editing works with the browser storage full' + (f' {errs}' if errs else ''))
    check(bool(__import__('re').search(r'storage|saved|kept', toast, __import__('re').I)), f'and says the tabs aren\'t being kept (toast: "{toast[:120]}")')
    ctx.close()
    browser.close()

print(f'\n{len(fails)} failure(s)')
for f in fails:
    print(' -', f)
sys.exit(1 if fails else 0)
