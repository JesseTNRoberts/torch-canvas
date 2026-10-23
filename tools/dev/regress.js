// writes the generated code of every example (both styles) to a folder, for before/after comparison
const path = require('path'), fs = require('fs'), NN = require(path.join(__dirname, '../../core.js'));
const dir = process.argv[2]; fs.mkdirSync(dir, { recursive: true });
const ex = path.join(__dirname, '../../examples/networks/');
for (const f of fs.readdirSync(ex)) {
  const src = fs.readFileSync(ex + f, 'utf8');
  const g = f.endsWith('.json') ? JSON.parse(src).graph : NN.importPython(src).graph;
  for (const lit of [true, false]) fs.writeFileSync(path.join(dir, f + '.' + lit + '.py'), NN.generate(g, { literalDims: lit, shapeComments: true }).code);
}
