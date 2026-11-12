# Opens the built app (index.html) in headless Chromium and checks what a person would hit first:
#   - it starts with no script errors, and opens every example network and library block, draws it, writes its code,
#     and plots its variance without errors
#   - it still starts when what it saved in the browser is damaged (bad JSON, wrong shapes, half-written graphs)
#   - text from pasted code (strings, comments, a category line) is shown as text, never run as HTML
#   - a full browser storage doesn't stop editing, and the person is told their tabs aren't being kept
#   - links that carry a model (#code=, #zip=, #url=) and the page interface for agents (window.torchCanvas) work, and
#     a graph in a link can't put HTML on the page
#
#   npm run build && python tools/check_browser.py [--quick]
# Needs playwright (pip install playwright; python -m playwright install chromium, or set CHROMIUM to a browser).
import http.server, json, os, sys, pathlib, threading, urllib.parse
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = (ROOT / 'index.html').as_uri()
QUICK = '--quick' in sys.argv
# examples whose variance plot is known not to work yet, and the message it gives instead
VARIANCE_UNSUPPORTED = {'networks/10_qwen2_5_vl_3b.json': "Can't predict through vision_window_index",
                        'networks/26_lif.json': "Can't predict through spike", 'networks/27_lif_axonal.json': "Can't predict through spike",
                        'networks/28_lif_synaptic.json': "Can't predict through spike", 'networks/29_lif_recurrent.json': "Can't predict through spike",
                        'networks/30_lif_adaptive.json': "Can't predict through spike"}
fails = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what)
    if not cond:
        fails.append(what)


def new_page(browser, storage=None, url=None):
    ctx = browser.new_context(viewport={'width': 1600, 'height': 1000})
    if storage is not None:  # what an earlier visit left in localStorage (seeded once: a reload keeps what the app saved;
        # window.name marks it, as the page's own storage can't be relied on this early on a reload)
        ctx.add_init_script('(() => { if (location.protocol !== "file:" || window.name === "seeded") return; window.name = "seeded"; localStorage.clear(); '
                            + ''.join(f'localStorage.setItem({k!r}, {v!r});' for k, v in storage.items()) + '})()')
    pg = ctx.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    # the only dialog the app should open is the name prompt for a saved view (answered "Exploded")
    pg.on('dialog', lambda d: d.accept('Exploded') if d.type == 'prompt' and 'view' in d.message else (errs.append('dialog: ' + d.message), d.dismiss()))
    pg.goto(url or URL)
    pg.wait_for_function('() => window.__tc && window.__tc.S && window.__tc.S.g', timeout=15000)
    return ctx, pg, errs


exe = os.environ.get('CHROMIUM') or ('/opt/pw-browsers/chromium' if os.path.exists('/opt/pw-browsers/chromium') else None)
with sync_playwright() as p:
    browser = p.chromium.launch(**({'executable_path': exe} if exe else {}))

    # ── every example ──
    ctx, pg, errs = new_page(browser)
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

    # ── Explode blocks and saved views ──
    ctx, pg, errs = new_page(browser)
    for f in [e['file'] for e in entries if e['kind'] == 'network']:
        n = pg.evaluate("async (f) => { const t = window.__tc; await t.openLibrary(t.S.library.find(x => x.file === f)); return t.S.root.nodes.some(n => n.type === 'block'); }", f)
        if not n:
            continue
        before = len(errs)
        res = pg.evaluate("() => { const t = window.__tc; t.explodeModel(); return [t.S.root.name, !!t.S.root.derived, t.S.root.nodes.length, document.querySelectorAll('svg [data-node]').length, /Could not generate/.test(document.querySelector('#code').textContent)]; }")
        check(res[1] and res[2] > 0 and res[3] > 0 and not res[4] and len(errs) == before, f'{f}: Explode blocks opens {res[0]} ({res[2]} layers)' + (f' {errs[before:]}' if len(errs) > before else ''))
    # draw out, fold back, save as a view, reopen it where it was left
    pg.evaluate("async () => { const t = window.__tc; await t.openLibrary(t.S.library.find(x => x.file === 'networks/07_gpt2.json')); t.explodeModel(); }")
    kept = pg.evaluate("() => window.__tc.S.root.nodes.filter(n => n.type === 'block').map(n => n.params.name)")
    pg.evaluate("(nm) => window.__tc.rederive({ [nm]: true })", kept[0])
    drawn = pg.evaluate("() => window.__tc.S.root.nodes.filter(n => n.type === 'block').length")
    pg.evaluate("(nm) => window.__tc.rederive({ [nm]: false })", kept[0])
    folded = pg.evaluate("() => window.__tc.S.root.nodes.filter(n => n.type === 'block').length")
    check(drawn == 0 and folded == 1, f'draw out and fold back ({kept[0]}: {drawn} blocks drawn out, then {folded} folded)')
    pg.evaluate("(nm) => window.__tc.rederive({ [nm]: undefined })", kept[0])
    pg.evaluate("() => { const t = window.__tc, n = t.S.g.nodes.find(n => n.type === 'linear'); n.x = 4321; n.y = -1234; t.render(); }")
    pg.click('#zView'); pg.click('[data-act="vsave"]')
    pg.click('#zView'); pg.click('[data-act="vback"]')
    views = pg.evaluate("() => Object.keys(window.__tc.S.root.views || {})")
    pg.evaluate("() => window.__tc.explodeModel()")
    back = pg.evaluate("() => window.__tc.S.g.nodes.some(n => n.x === 4321 && n.y === -1234)")
    pg.reload(); pg.wait_for_function('() => window.__tc && window.__tc.S.g')
    kept_views = pg.evaluate("() => window.__tc.S.docs.map(d => (d.id === window.__tc.S.cur ? window.__tc.S.root : d.root)).some(r => r.views && r.views.Exploded)")
    check(views == ['Exploded'] and back and kept_views and not errs, f'a saved exploded view reopens as it was saved, and is kept (views {views})' + (f' {errs}' if errs else ''))
    # an arrangement of a model saved as a view comes back after the layers have been moved
    pg.evaluate("async () => { const t = window.__tc; await t.openLibrary(t.S.library.find(x => x.file === 'networks/01_residual_mlp.json')); }")
    pg.click('#zView'); pg.click('[data-act="vsave"]')
    name = pg.evaluate("() => Object.keys(window.__tc.S.root.views || {})[0]")
    pos0 = pg.evaluate("() => window.__tc.S.g.nodes.map(n => [n.x, n.y])")
    pg.evaluate("() => { const t = window.__tc; t.S.g.nodes.forEach(n => { n.x += 50; }); t.render(); }")
    pg.evaluate("(nm) => window.__tc.applyView(nm)", name)
    check(pg.evaluate("() => window.__tc.S.g.nodes.map(n => [n.x, n.y])") == pos0 and not errs, f'an arrangement saved as a view ({name}) comes back')
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
            # what could be saved was: the repaired tab is open (seeding took effect)
            if STORE in storage and '"name":"M"' in storage[STORE] and what != 'block use of a missing block':
                name = pg.evaluate("() => window.__tc.S.root.name")
                if name != 'M':
                    errs.append(f'the repaired tab M was not reopened (showing {name})')
        except Exception as ex:
            started = False
            errs.append(str(ex).splitlines()[0])
        check(started and not errs, f'starts with damaged saved data ({what})' + (f': {errs[:2]}' if errs else ''))
        if ctx:
            ctx.close()

    # ── tabs come back after a reload: an untouched example (kept as a reference) and an edited model ──
    ctx, pg, errs = new_page(browser)
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

    # ── …and when the storage is too full to copy it, the saved data is left exactly as it was ──
    ctx = browser.new_context(viewport={'width': 1600, 'height': 1000})
    saved_doc = '{"v":2,"docs":[{"name":"Kept","symbols":{},"nodes":[],"edges":[]}],"from":[null],"pad":"' + 'x' * 200000 + '"}'
    ctx.add_init_script('(() => { if (location.protocol !== "file:" || window.name === "seeded") return; window.name = "seeded"; localStorage.clear(); '
                        f'localStorage.setItem("torchcanvas.graph.v1", {saved_doc!r}); localStorage.setItem("torchcanvas.booting", "{old}"); '
                        'let i = 0; for (let size = 1 << 20; size >= 1; size >>= 1) { const s = "x".repeat(size); try { for (;;) localStorage.setItem("filler" + i++, s); } catch (e) {} } })()')
    pg = ctx.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_function('() => window.__tc && window.__tc.S.g'); pg.wait_for_timeout(700)
    same = pg.evaluate("(s) => localStorage.getItem('torchcanvas.graph.v1') === s", saved_doc)
    pg.evaluate("() => { const t = window.__tc; t.S.sel.nodes.clear(); t.S.sel.nodes.add(t.S.g.nodes[1].id); t.selectionChanged(); }")
    pg.keyboard.press('Delete'); pg.wait_for_timeout(300)
    still = pg.evaluate("(s) => localStorage.getItem('torchcanvas.graph.v1') === s", saved_doc)
    msg = pg.evaluate("() => document.querySelector('#toast').textContent")
    check(same and still and not errs, 'leaves saved data it can neither open nor copy untouched, even after edits' + (f' {errs}' if errs else ''))
    ctx.close()

    # ── the tab that was showing comes back, even when an earlier saved tab is dropped ──
    good = lambda nm: '{"name":"%s","symbols":{"e":8},"nodes":[{"id":"a","type":"input","x":0,"y":0,"params":{"name":"x","shape":"e, 4"}}],"edges":[],"groups":[]}' % nm
    ctx, pg, errs = new_page(browser, {STORE: '{"v":2,"docs":[{"nodes":"broken"},%s,%s],"from":[null,null,null],"cur":2}' % (good('First'), good('Second'))})
    cur = pg.evaluate("() => window.__tc.S.root.name")
    check(cur == 'Second' and not errs, f'reopens the tab that was showing ({cur})')
    ctx.close()

    # ── text from code is text ──
    ctx, pg, errs = new_page(browser)
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

    # ── links that carry a model, and the page interface for agents ──
    tiny = 'import torch\nimport torch.nn as nn\n\n\nclass Tiny(nn.Module):\n    def __init__(self, d=32, h=64):\n        super().__init__()\n        self.fc1 = nn.Linear(d, h)\n        self.fc2 = nn.Linear(h, d)\n\n    def forward(self, x):\n        return x + self.fc2(torch.relu(self.fc1(x)))\n'
    ctx, pg, errs = new_page(browser, url=URL + '#code=' + urllib.parse.quote(tiny, safe=''))
    pg.wait_for_timeout(600)
    tabs, hsh, sm = pg.evaluate("() => [torchCanvas.tabs(), location.hash, torchCanvas.summary()]")
    check([t['name'] for t in tabs] == ['Tiny'] and hsh == '' and sm['params'] == 4192 and sm['outputs'][0]['shape'][-1] == 32,
          f'a #code= link opens the model on its own, and the link is cleared ({[t["name"] for t in tabs]}, {sm["params"]} params)')
    # the links Copy link makes open the same model, and the one with the drawing keeps where everything sits
    pg.evaluate("() => torchCanvas.openExample('gpt2')"); pg.wait_for_timeout(300)
    pg.evaluate("() => { const g = window.__tc.S.g; g.nodes[0].x += 37; window.__tc.render(); }")
    want = pg.evaluate("() => [torchCanvas.code(), window.__tc.S.g.nodes.map(n => [n.x, n.y])]")
    links = {layout: pg.evaluate("(o) => torchCanvas.link(o)", {'layout': layout}) for layout in (False, True)}
    for layout, link in links.items():
        pg.evaluate("(h) => { location.hash = h; }", link[link.index('#'):]); pg.wait_for_timeout(700)
        got = pg.evaluate("() => [torchCanvas.code(), window.__tc.S.g.nodes.map(n => [n.x, n.y]), torchCanvas.tabs().length]")
        check(got[0] == want[0] and (not layout or got[1] == want[1]), f'the {"drawing" if layout else "code"} link of GPT-2 ({len(link):,} characters) opens the same model' + (' where it was drawn' if layout else ''))
    # Share: an untouched example gets its short ?open= link, an edited model a link that carries the drawing
    pg.evaluate("() => torchCanvas.openExample('gpt2')"); pg.wait_for_timeout(300)
    short = pg.evaluate("() => torchCanvas.shareLink()")
    check(short.endswith('?open=gpt2'), f'Share on an unchanged example gives its short link ({short[-24:]})')
    pg.evaluate("() => { location.hash = '#code=' + encodeURIComponent(%s); }" % json.dumps(tiny)); pg.wait_for_timeout(600)
    pg.evaluate("() => { const g = window.__tc.S.g; g.nodes[0].x += 41; window.__tc.render(); }")
    want = pg.evaluate("() => [torchCanvas.code(), window.__tc.S.g.nodes.map(n => [n.x, n.y])]")
    pg.evaluate("() => { window.__copied = null; navigator.clipboard.writeText = async t => { window.__copied = t; }; document.querySelector('#bShare').click(); }"); pg.wait_for_timeout(400)
    link = pg.evaluate("() => window.__copied") or ''
    ok = '#zip=' in link
    if ok:
        pg.evaluate("(h) => { location.hash = h; }", link[link.index('#'):]); pg.wait_for_timeout(700)
        ok = pg.evaluate("() => [torchCanvas.code(), window.__tc.S.g.nodes.map(n => [n.x, n.y])]") == want
    check(ok, f'the Share button copies a link ({len(link):,} characters) that opens the model where it was drawn')
    pg.evaluate("() => { location.hash = '#code=' + encodeURIComponent('class X(nn.Module):\\n  def forward(self, x:\\n'); }"); pg.wait_for_timeout(400)
    shown = pg.evaluate("() => [!document.querySelector('#importModal').hidden, document.querySelector('#imErr').textContent]")
    check(shown[0] and 'Line' in shown[1], f'code in a link that can\'t be read opens Import with the message ({shown[1][:60]})')
    # #url= fetches the file (here from a local server that allows it, as raw.githubusercontent.com and gists do)
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k): super().__init__(*a, directory=str(ROOT / 'examples' / 'networks'), **k)
        def end_headers(self): self.send_header('Access-Control-Allow-Origin', '*'); super().end_headers()
        def log_message(self, *a): pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    pyfile = sorted(f.name for f in (ROOT / 'examples' / 'networks').glob('*.py'))[0]
    n0 = pg.evaluate("() => torchCanvas.tabs().length")
    pg.evaluate("(u) => { location.hash = '#url=' + encodeURIComponent(u); }", f'http://127.0.0.1:{srv.server_address[1]}/{pyfile}'); pg.wait_for_timeout(900)
    n1, code1 = pg.evaluate("() => [torchCanvas.tabs().length, torchCanvas.code()]")
    missing = pg.evaluate("(u) => torchCanvas.loadUrl(u).then(() => 'opened', e => e.message)", f'http://127.0.0.1:{srv.server_address[1]}/no_such_file.py')
    check(n1 == n0 + 1 and 'nn.Module' in code1 and missing.startswith("Couldn't fetch"), f'a #url= link opens {pyfile}, and a missing file says so ({missing[:50]})')
    srv.shutdown()
    # a graph in a link is checked: any text in it stays text, and a setting outside its choices goes back to its default
    res = pg.evaluate("""async () => {
      const x = '<img src=x onerror="window.__pwned=(window.__pwned||[]).concat(window.__cur)">', done = new Set(), crash = [];
      for (const e of torchCanvas.examples().filter(e => e.kind === 'network')) {
        torchCanvas.openExample(e.file); const base = torchCanvas.graph(); await torchCanvas.close(torchCanvas.tabs().find(t => t.active).id);
        const walk = (g, f) => { f(g); for (const b of Object.values(g.blocks || {})) walk(b.graph, f); };
        const fields = new Set(); walk(base, g => g.nodes.forEach(n => { for (const k of Object.keys(n.params)) if (typeof n.params[k] === 'string') fields.add(n.type + '.' + k); }));
        for (const fld of fields) {
          if (done.has(fld)) continue; done.add(fld); window.__cur = fld;
          const g = JSON.parse(JSON.stringify(base)), [t, k] = fld.split('.');
          g.name = x; walk(g, gg => { gg.nodes.forEach(n => { if (n.type === t) { n.params[k] = x; n.params.note = x; } }); (gg.groups || []).forEach(gr => { gr.label = x; }); });
          const r = await torchCanvas.load(JSON.stringify(g), { quiet: true }); if (!r.ok) { crash.push(fld + ': ' + r.error); continue; }
          const tc = window.__tc;
          for (const n of tc.S.g.nodes.filter(n => n.type === t).slice(0, 2)) { try { tc.S.sel.nodes.clear(); tc.S.sel.nodes.add(n.id); tc.S.drawerOpen = true; tc.selectionChanged(); } catch (er) { crash.push(fld + ': ' + er.message); } }
          await new Promise(r => setTimeout(r, 40));
          await torchCanvas.close(torchCanvas.tabs().find(t => t.active).id);
        }
      }
      return { n: done.size, pwned: [...new Set(window.__pwned || [])], crash, img: !!document.querySelector('img[src=x]') };
    }""")
    check(not res['pwned'] and not res['img'] and not res['crash'], f'text in every setting of a linked graph stays text ({res["n"]} settings)' + (f' pwned by {res["pwned"]}' if res['pwned'] else '') + (f' {res["crash"][:3]}' if res['crash'] else ''))
    check(not errs, 'no script errors with links' + (f' {errs}' if errs else ''))
    ctx.close()

    # ── full storage ──
    ctx, pg, errs = new_page(browser)
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

    # ── Spike: a double-click opens its larger view in place (the step forward, the sigmoid backward) ──
    ctx, pg, errs = new_page(browser)
    pg.evaluate("async () => { const t = window.__tc; await t.openLibrary(t.S.library.find(e => e.file === 'networks/26_lif.json')); }")
    pg.wait_for_timeout(400)
    sid = pg.evaluate("() => window.__tc.S.g.nodes.find(n => n.params.fn === 'spike').id")
    node, view = pg.locator(f'[data-node="{sid}"]'), "() => { const v = document.querySelector('[data-spikeview]'); return v ? v.querySelectorAll('.sv-card').length + ' ' + v.querySelectorAll('.sv-step').length + ' ' + v.querySelectorAll('.sv-sig').length : ''; }"
    check(pg.evaluate(f"() => document.querySelectorAll('[data-node=\"{sid}\"] .sv-step, [data-node=\"{sid}\"] .sv-sig').length") == 2, 'a Spike node shows a step over a sigmoid')
    n0 = pg.evaluate("() => JSON.stringify(window.__tc.S.g)")
    node.dblclick(); pg.wait_for_timeout(300)
    check(pg.evaluate(view) == '2 1 1', f'a double-click opens its view: a Forward card with the step, a Backward card with the sigmoid ({pg.evaluate(view)!r})')
    clear = pg.evaluate("""() => { const v = document.querySelector('[data-spikeview] .n-box').getBoundingClientRect(); let hit = [];
      for (const el of document.querySelectorAll('#gNodes [data-node]')) { if (el.querySelector('[data-spikeview]')) continue; const r = el.getBoundingClientRect();
        if (r.left < v.right && r.right > v.left && r.top < v.bottom && r.bottom > v.top) hit.push(el.dataset.node); }
      return hit; }""")
    check(not clear, f'the rest of the diagram is drawn moved apart, so the open Spike covers no other node {clear}')
    check(pg.evaluate("() => JSON.stringify(window.__tc.S.g)") == n0, 'and the positions in the model are as they were')
    pg.locator('[data-spikeview] .sv-card').first.click(); pg.wait_for_timeout(150)
    check(pg.evaluate(view) != '', 'a click inside the view leaves it open')
    pg.locator('[data-spikeclose]').click(); pg.wait_for_timeout(150)
    check(pg.evaluate(view) == '', 'its arrow closes it')
    node.dblclick(); pg.wait_for_timeout(300); pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
    check(pg.evaluate(view) == '', 'Esc closes it')
    pg.locator(f'[data-openspike="{sid}"]').click(); pg.wait_for_timeout(300)
    check(pg.evaluate(view) == '2 1 1', 'the open mark on its corner opens it with one click')
    other = pg.evaluate("() => window.__tc.S.g.nodes.find(n => n.type === 'linear').id")
    pg.locator(f'[data-node="{other}"]').click(); pg.wait_for_timeout(200)
    check(pg.evaluate(view) == '' and pg.evaluate("() => JSON.stringify(window.__tc.S.g)") == n0, 'a press on another node closes it and moves nothing')
    check(pg.evaluate("() => JSON.stringify(window.__tc.S.g)") == n0 and not errs, 'opening and closing the view changes nothing in the model' + (f' {errs}' if errs else ''))
    ctx.close()
    browser.close()

print(f'\n{len(fails)} failure(s)')
for f in fails:
    print(' -', f)
    if os.environ.get('GITHUB_ACTIONS'):  # an annotation on the run's summary page
        print('::error title=browser check::' + f.replace('%', '%25').replace('\n', '%0A')[:900])
sys.exit(1 if fails else 0)
