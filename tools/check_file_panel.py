# The code panel with a .py file open, in headless Chromium with a stand-in for the file (the File System Access
# API is stubbed, so nothing is read from or written to disk): the panel shows the file's own text, diagram edits
# are written into it, Ctrl+Z, Save, the save toast's Undo, edits typed in the panel, Save as, and closing the file.
#
#   npm run build && python tools/check_file_panel.py
import pathlib, sys
from playwright.sync_api import sync_playwright
URL = (pathlib.Path(__file__).resolve().parent.parent / 'index.html').as_uri()
SRC = '''import torch
import torch.nn as nn


class Net(nn.Module):
    """Tiny net."""

    def __init__(self, n_in=8, n_out=4):
        super().__init__()
        # keep me
        self.fc1 = nn.Linear(n_in, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, n_out)

    def forward(self, x):
        """Doc."""
        return self.fc2(self.act(self.fc1(x)))

    def predict(self, x):
        return self.forward(x).argmax(1)


if __name__ == "__main__":
    model = Net()
    print(model(torch.randn(8, 8)).shape)
'''
STUB = """
const mk = (name, disk) => ({ kind: 'file', name,
  getFile: async () => ({ name, lastModified: disk.lm, text: async () => disk.text }),
  createWritable: async () => { let buf = ''; return { write: async t => { buf += t; }, close: async () => { disk.text = buf; disk.lm += 1000; disk.writes++; } }; },
  queryPermission: async () => 'granted', requestPermission: async () => 'granted', isSameEntry: async function (o) { return o === this; } });
window.__disk = { text: %r, lm: 1000, writes: 0 }; window.__disk2 = { text: '', lm: 1, writes: 0 };
window.__h = mk('net.py', window.__disk);
window.showOpenFilePicker = async () => [window.__h];
window.showSaveFilePicker = async () => mk('copy.py', window.__disk2);
""" % SRC
fails = []
def check(c, what, extra=''):
    print(('ok   ' if c else 'FAIL ') + what + ((' :: ' + str(extra)) if (extra and not c) else ''))
    if not c: fails.append(what)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_context(viewport={'width':1500,'height':950}).new_page()
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('dialog', lambda d: (errs.append('dialog: ' + d.message), d.dismiss()))
    pg.add_init_script(STUB)
    pg.goto(URL); pg.wait_for_timeout(1000)
    panel = lambda: pg.evaluate("() => [...document.querySelectorAll('#code .ln')].map(l => l.textContent === ' ' ? '' : l.textContent).join('\\n')")
    pill = lambda: pg.evaluate("() => (document.querySelector('#fileBar .pill') || {}).textContent")
    disk = lambda: pg.evaluate("() => window.__disk.text")
    toasts = lambda: pg.evaluate("() => [...document.querySelectorAll('[class*=toast]')].map(t => t.textContent).join(' | ')")
    fc1 = lambda: pg.evaluate("() => { const n = window.__tc.S.root.nodes.find(z => z.type === 'linear' && z.params.name === 'fc1'); return n.id + ':' + n.params.out; }")
    def resize(v):
        pg.locator(f'[data-node="{fc1().split(":")[0]}"]').dblclick(); pg.wait_for_timeout(300)
        f = pg.locator('#insp #f-wsize'); f.fill(v); f.press('Tab'); pg.wait_for_timeout(400)
    pg.keyboard.press('Control+o'); pg.wait_for_timeout(900)
    check(panel().rstrip() == SRC.rstrip(), 'a. the panel shows the file as it is on disk', panel()[:300])
    check(pill() == 'saved', 'a. pill says saved', pill())
    pg.locator(f'[data-node="{fc1().split(":")[0]}"]').click(); pg.wait_for_timeout(300)
    hl = pg.evaluate("() => [...document.querySelectorAll('#code .ln.hl')].map(l => l.textContent.trim())")
    check(hl == ['self.fc1 = nn.Linear(n_in, 16)', 'return self.fc2(self.act(self.fc1(x)))'], "b. selecting fc1 highlights every line that touches it: where it is made and where it is used", hl)
    relu = pg.evaluate("() => window.__tc.S.root.nodes.find(z => z.type === 'act').id")
    pg.evaluate("(id) => { const S = window.__tc.S; S.sel.nodes.clear(); S.sel.nodes.add(id); window.__tc.selectionChanged(); }", relu); pg.wait_for_timeout(200)
    hl = pg.evaluate("() => [...document.querySelectorAll('#code .ln.hl')].map(l => l.textContent.trim())")
    check(hl == ['self.act = nn.ReLU()', 'return self.fc2(self.act(self.fc1(x)))'], "b. an activation written as a module highlights self.act = nn.ReLU() and its use", hl)
    resize('32')
    t = panel()
    check('self.fc1 = nn.Linear(n_in, 32)' in t and 'self.fc2 = nn.Linear(32, n_out)' in t and '# keep me' in t and 'def predict' in t and '"""Doc."""' in t and 'self.act(self.fc1(x))' in t, 'c. a resize shows in the file text, everything hand-written still there', t)
    check(pill() == 'unsaved changes', 'c. pill says unsaved changes', pill())
    check(disk() == SRC, 'c. nothing written yet')
    pg.locator('#wrap').click(position={'x': 30, 'y': 400}); pg.keyboard.press('Control+z'); pg.wait_for_timeout(400)
    check(panel().rstrip() == SRC.rstrip() and pill() == 'saved', 'd. Ctrl+Z puts the text back and the pill says saved', pill())
    resize('32'); pg.keyboard.press('Control+s'); pg.wait_for_timeout(700)
    saved1 = disk()
    check(saved1.rstrip() == panel().rstrip() and 'nn.Linear(n_in, 32)' in saved1 and '# keep me' in saved1 and pill() == 'saved', 'e. Ctrl+S writes exactly what the panel shows', pill())
    pg.locator('[class*=toast] button', has_text='Undo').first.click(); pg.wait_for_timeout(700)
    check(disk() == SRC, 'e. Undo on the toast restores the file')
    check(fc1().endswith(':32') and pill() == 'unsaved changes' and 'nn.Linear(n_in, 32)' in panel(), 'e. after Undo the diagram keeps the change and it shows as unsaved', (fc1(), pill()))
    check('unsaved' in toasts(), 'e. the toast says so', toasts())
    pg.keyboard.press('Control+s'); pg.wait_for_timeout(700)
    check(disk() == saved1 and pill() == 'saved', 'e. saving again writes it back')
    # f. edit the text: add a comment only
    pg.locator('#bEditCode').click(); pg.wait_for_timeout(300)
    ta_val = pg.evaluate("() => document.querySelector('#codeTa').value")
    check(ta_val.rstrip() == saved1.rstrip(), "f. the editor starts from the file's text")
    pg.evaluate("() => { const ta = document.querySelector('#codeTa'); ta.value = ta.value.replace('        # keep me', '        # keep me\\n        # typed in the panel'); ta.dispatchEvent(new Event('input')); }")
    pg.locator('#bCodeApply').click(); pg.wait_for_timeout(500)
    check('# typed in the panel' in panel() and pill() == 'unsaved changes', 'f. an applied comment is in the panel and counts as unsaved', pill())
    pg.keyboard.press('Control+s'); pg.wait_for_timeout(700)
    check('# typed in the panel' in disk() and 'def predict' in disk() and pill() == 'saved', 'f. and is saved', pill())
    # g. edit the text: change a size
    pg.locator('#bEditCode').click(); pg.wait_for_timeout(300)
    pg.evaluate("() => { const ta = document.querySelector('#codeTa'); ta.value = ta.value.replace('nn.Linear(n_in, 32)', 'nn.Linear(n_in, 48)').replace('nn.Linear(32, n_out)', 'nn.Linear(48, n_out)'); ta.dispatchEvent(new Event('input')); }")
    pg.locator('#bCodeApply').click(); pg.wait_for_timeout(500)
    check(fc1().endswith(':48') and 'nn.Linear(n_in, 48)' in panel() and '# typed in the panel' in panel(), 'g. a size typed in the text reaches the diagram, text kept as typed', fc1())
    # h. a change to the wiring: forward is written again, the rest kept
    pg.evaluate("""() => { const S = window.__tc.S, g = S.root; const o = g.nodes.find(n => n.type === 'output'), e = g.edges.find(z => z.to === o.id);
      const n = { id: 'zzrelu', type: 'act', x: o.x, y: o.y + 50, params: { fn: 'tanh', dim: '-1', title: '', var: '' } }; g.nodes.push(n); const port = e.port; e.to = n.id; e.port = 0; g.edges.push({ id: 'zzedge', from: n.id, to: o.id, port }); }""")
    pg.locator('#wrap').click(position={'x': 30, 'y': 400}); resize('64')
    t = panel()
    check('tanh' in t and '"""Doc."""' in t and 'def predict' in t and '# keep me' in t and 'nn.Linear(n_in, 64)' in t, 'h. a new activation rewrites forward only; docstring, comment and predict kept', t)
    # i. save as
    pg.locator('#fbSaveAs').click(); pg.wait_for_timeout(700)
    t2 = pg.evaluate("() => window.__disk2.text")
    check('# keep me' in t2 and 'def predict' in t2 and 'tanh' in t2, "i. Save as writes the file's text with the changes, not regenerated code", t2[:300])
    check(pill() == 'saved' and panel().rstrip() == t2.rstrip(), 'i. the new file is open and saved', pill())
    # j. close the file
    pg.locator('#fbClose').click(); pg.wait_for_timeout(400)
    check('# keep me' not in panel() and 'class Net' in panel(), 'j. with the file closed the panel shows generated code again')
    check(not errs, 'no page errors', errs)
    b.close()
print(len(fails), 'failure(s)')
sys.exit(1 if fails else 0)
