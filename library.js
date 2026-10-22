// Torch Canvas library: every file in examples/networks and examples/blocks becomes a built-in entry.
// A .py file is any nn.Module; a .json file is a diagram saved from Torch Canvas ({ title, description, graph }).
// Metadata for .py files comes from comment lines at the top: "# title: …", "# description: …", "# labels: h=hidden".
const fs = require('fs');
const path = require('path');

// A network file whose name starts with this prefix is ignored (e.g. _scratch.json), so it can stay in the folder
// without showing up under Open / import.
const HIDDEN = '_';
function readEntry(file, kind) {
  const source = fs.readFileSync(file, 'utf8');
  const ext = path.extname(file).toLowerCase();
  const entry = { kind, file: path.relative(path.join(__dirname, 'examples'), file).split(path.sep).join('/'), type: ext === '.json' ? 'json' : 'py', source };
  if (entry.type === 'json') {
    try { const j = JSON.parse(source); entry.title = j.title; entry.description = j.description; if (j.labels) entry.labels = j.labels; } catch { return null; }
  } else {
    const meta = {};
    for (const line of source.split('\n')) { const m = /^#\s*(title|description|labels)\s*:\s*(.*)$/i.exec(line); if (m) meta[m[1].toLowerCase()] = m[2].trim(); else if (line.trim() && !line.startsWith('#')) break; }
    Object.assign(entry, meta);
  }
  entry.title = entry.title || path.basename(file, ext).replace(/^\d+_/, '').replace(/_/g, ' ');
  return entry;
}
function scanLibrary(root = path.join(__dirname, 'examples')) {
  const out = [];
  for (const kind of ['networks', 'blocks']) {
    const dir = path.join(root, kind);
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir).sort()) {
      if (!/\.(py|json)$/i.test(f)) continue;
      if (kind === 'networks' && f.startsWith(HIDDEN)) continue; // kept in the folder, left out of the app
      const e = readEntry(path.join(dir, f), kind === 'networks' ? 'network' : 'block');
      if (e) out.push(e);
    }
  }
  return out;
}
module.exports = { scanLibrary, HIDDEN };
