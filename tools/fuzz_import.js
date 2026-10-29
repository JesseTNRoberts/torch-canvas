// Fuzzes the code reader: example code with random damage (lines deleted, duplicated or swapped, text cut short,
// tokens replaced, brackets dropped, junk inserted) must either read or fail with an ordinary Error, quickly, and
// whatever reads must generate code without throwing. Catches crashes and hangs on code people paste.
//   node tools/fuzz_import.js [iterations=2000] [seed=1]
const fs = require('fs'), path = require('path'), NN = require('../core.js');
function fuzz(iterations = 2000, seed = 1, log = console.log) {
  let s = seed >>> 0 || 1; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296), pick = a => a[Math.floor(rnd() * a.length)];
  const ex = path.join(__dirname, '..', 'examples', 'networks'), srcs = [];
  for (const f of fs.readdirSync(ex).sort()) {
    const t = fs.readFileSync(path.join(ex, f), 'utf8'), g = f.endsWith('.json') ? JSON.parse(t).graph : NN.importPython(t).graph;
    srcs.push([f, NN.generate(g, { literalDims: true }).code], [f + ' (named)', NN.generate(g, { literalDims: false }).code]);
  }
  const TOK = ['(', ')', '[', ']', ':', ',', '=', '+', '*', '.', 'self', 'x', 'return', 'def', 'class', 'for', 'in', 'nn.Linear', 'F.relu', '0', '-1', '"', "'", '\\', '#', '\t', '    ', '\n', 'None', 'lambda', '**', '@', 'torch.cat', 'yield', '...'];
  const damage = [
    t => { const L = t.split('\n'); L.splice(Math.floor(rnd() * L.length), 1); return L.join('\n'); },
    t => { const L = t.split('\n'), i = Math.floor(rnd() * L.length); L.splice(i, 0, L[i]); return L.join('\n'); },
    t => { const L = t.split('\n'), i = Math.floor(rnd() * L.length), j = Math.floor(rnd() * L.length); [L[i], L[j]] = [L[j], L[i]]; return L.join('\n'); },
    t => t.slice(0, Math.floor(rnd() * t.length)),
    t => { const i = Math.floor(rnd() * t.length); return t.slice(0, i) + pick(TOK) + t.slice(i); },
    t => { const m = [...t.matchAll(/[()\[\]]/g)]; if (!m.length) return t; const i = pick(m).index; return t.slice(0, i) + t.slice(i + 1); },
    t => t.replace(/\b\d+\b/, () => pick(['0', '-3', '1e9', '2.5', 'n', 'x'])),
    t => t.replace(/\b(nn\.\w+|F\.\w+|torch\.\w+)/, () => pick(['nn.Foo', 'F.bar', 'torch.baz', 'nn.Linear', 'nn.Conv2d', 'getattr'])),
  ];
  let ok = 0, rejected = 0, slow = 0; const fails = [];
  for (let k = 0; k < iterations; k++) {
    const [f, base] = pick(srcs); let t = base; const n = 1 + Math.floor(rnd() * 3);
    for (let j = 0; j < n; j++) t = pick(damage)(t);
    const t0 = Date.now();
    try {
      const r = NN.importPython(t);
      try { NN.generate(r.graph, { literalDims: true }); NN.inferShapes(r.graph); ok++; }
      catch (e) { if (!(e instanceof Error)) throw e; fails.push(`${f}: reading worked but generating threw ${e.message}`); }
    } catch (e) {
      if (e instanceof Error && !(e instanceof RangeError) && !(e instanceof TypeError) && !(e instanceof ReferenceError)) rejected++;
      else fails.push(`${f}: ${e && e.constructor && e.constructor.name}: ${e && e.message}\n--- code ---\n${t.slice(0, 4000)}`);
    }
    const dt = Date.now() - t0; if (dt > 3000) { slow++; fails.push(`${f}: took ${dt} ms`); }
  }
  log(`fuzz: ${iterations} damaged files: ${ok} read, ${rejected} refused with a message, ${fails.length} problems${slow ? ` (${slow} slow)` : ''}`);
  return fails;
}
if (require.main === module) {
  const fails = fuzz(+process.argv[2] || 2000, +process.argv[3] || 1);
  for (const f of fails.slice(0, 10)) console.log('\n' + f);
  process.exit(fails.length ? 1 : 0);
}
module.exports = fuzz;
