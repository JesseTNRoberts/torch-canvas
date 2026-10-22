// Counts routing problems in every example network and library block:
//   overlaps: two connections from different sources running along the same line (collinear, overlapping > 6px)
//   through:  a connection segment passing through a box that is neither its source nor its target
// usage: node tools/route_check.js [--shots dir]
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const shots = process.argv.includes('--shots') ? process.argv[process.argv.indexOf('--shots') + 1] : null;
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  await p.route(/fonts\.googleapis/, r => r.abort());
  await p.goto('file://' + path.join(__dirname, '..', 'index.html'));
  await p.waitForTimeout(500);
  const files = await p.evaluate(() => window.__tc.S.library.map(x => x.file));
  let tot = { overlaps: 0, through: 0 };
  for (const f of files) {
    const graphs = await p.evaluate(f => {
      const T = window.__tc; T.openLibrary(T.S.library.find(x => x.file === f));
      const root = T.S.root, keys = [null, ...Object.keys(root.blocks || {})];
      return keys.map(k => ({ k, name: k ? root.blocks[k].name : root.name }));
    }, f);
    for (const { k, name } of graphs) {
      const r = await p.evaluate(k => {
        const T = window.__tc; T.switchTo(k || 'model'); T.render();
        const g = T.S.g, segs = [];
        for (const e of g.edges) {
          const rt = T.route(e); if (!rt) continue; const P = rt.pts;
          for (let i = 1; i < P.length; i++) segs.push({ e, src: T.keyOf(e), a: P[i - 1], b: P[i], first: i === 1, last: i === P.length - 1 });
        }
        let overlaps = 0, through = 0; const bad = [];
        for (let i = 0; i < segs.length; i++) for (let j = i + 1; j < segs.length; j++) {
          const s = segs[i], t = segs[j]; if (s.src === t.src) continue;
          if (s.e.to === t.e.to && s.last && t.last) continue;
          const sv = Math.abs(s.a[0] - s.b[0]) < 0.5, tv = Math.abs(t.a[0] - t.b[0]) < 0.5;
          const sh = Math.abs(s.a[1] - s.b[1]) < 0.5, th = Math.abs(t.a[1] - t.b[1]) < 0.5;
          let ov = 0;
          if (sv && tv && Math.abs(s.a[0] - t.a[0]) < 2) ov = Math.min(Math.max(s.a[1], s.b[1]), Math.max(t.a[1], t.b[1])) - Math.max(Math.min(s.a[1], s.b[1]), Math.min(t.a[1], t.b[1]));
          if (sh && th && Math.abs(s.a[1] - t.a[1]) < 2) ov = Math.min(Math.max(s.a[0], s.b[0]), Math.max(t.a[0], t.b[0])) - Math.max(Math.min(s.a[0], s.b[0]), Math.min(t.a[0], t.b[0]));
          if (ov > 6) { overlaps++; bad.push(['ov', s.e.id, t.e.id]); }
        }
        for (const s of segs) for (const n of g.nodes) {
          if (n.id === s.e.from || n.id === s.e.to || (T.drawFrom && n.id === T.drawFrom(s.e).id)) continue;
          const q = T.box(n), x0 = Math.min(s.a[0], s.b[0]), x1 = Math.max(s.a[0], s.b[0]), y0 = Math.min(s.a[1], s.b[1]), y1 = Math.max(s.a[1], s.b[1]);
          if (x1 > q.x + 2 && x0 < q.x + q.w - 2 && y1 > q.y + 2 && y0 < q.y + q.h - 2) { through++; bad.push(['thr', s.e.id, n.id]); }
        }
        return { overlaps, through, n: g.edges.length };
      }, k);
      tot.overlaps += r.overlaps; tot.through += r.through;
      if (r.overlaps || r.through) console.log(`${f.padEnd(44)} ${(k ? name : '(network)').padEnd(28)} edges ${String(r.n).padStart(3)}  overlaps ${String(r.overlaps).padStart(3)}  through ${String(r.through).padStart(3)}`);
      if (shots) { await p.evaluate(() => window.__tc && document.querySelector('#zFit').click()); await p.waitForTimeout(100); await p.screenshot({ path: path.join(shots, (f + '_' + name).replace(/[^\w.]/g, '_') + '.png') }); }
    }
  }
  console.log('TOTAL overlaps', tot.overlaps, 'through', tot.through);
  await b.close();
})();
