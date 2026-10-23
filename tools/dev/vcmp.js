// Compares the variance-plot predictions of every example network in two examples folders (layer by layer).
// usage: node tools/dev/vcmp.js path/to/old/examples examples
const fs = require('fs'), E = require(require('path').join(__dirname, '../../engine.js'));
const A = process.argv[2], B = process.argv[3];
for (const f of fs.readdirSync(A + '/networks').filter(f => f.endsWith('.json'))) {
  const ga = JSON.parse(fs.readFileSync(A + '/networks/' + f)).graph, gb = JSON.parse(fs.readFileSync(B + '/networks/' + f)).graph;
  const ra = E.predict(ga), rb = E.predict(gb);
  if (ra.error || rb.error) { console.log(f, 'error', ra.error, '|', rb.error); continue; }
  const la = ra.layers.map(l => [l.fwdOut, l.bwdIn]), lb = rb.layers.map(l => [l.fwdOut, l.bwdIn]);
  let worst = 0; if (la.length !== lb.length) { console.log(f, 'layer count', la.length, lb.length); continue; }
  la.forEach((x, i) => x.forEach((v, j) => { const w = lb[i][j]; if (v || w) worst = Math.max(worst, Math.abs(v - w) / Math.max(Math.abs(v), Math.abs(w), 1e-12)); }));
  console.log(f.padEnd(32), la.length, 'layers', worst < 1e-9 ? 'same' : 'max rel diff ' + worst.toExponential(2));
}
