// Torch Canvas: tiny static server. Run `npm start`, then open http://localhost:5173 in Chrome.
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 5173;
const ROOT = __dirname;
const { scanLibrary } = require('./library.js');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.py': 'text/x-python', '.json': 'application/json' };

http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  // the library is read fresh on every request, so new files in examples/ appear without rebuilding
  if (url === '/examples/index.json') {
    try { res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); return res.end(JSON.stringify(scanLibrary())); }
    catch (e) { res.writeHead(500); return res.end(String(e)); }
  }
  // the app writes edited library blocks back to their files, deletes deleted ones, and rewrites example networks that used them
  if (req.method === 'DELETE') {
    if (!/^\/examples\/blocks\/[\w-][\w.-]*\.(py|json)$/.test(url)) { res.writeHead(403); return res.end(); }
    return fs.unlink(path.join(ROOT, url), err => { res.writeHead(err ? 404 : 204); res.end(); });
  }
  if (req.method === 'PUT') {
    if (!/^\/examples\/(blocks|networks)\/[\w-][\w.-]*\.(py|json)$/.test(url)) { res.writeHead(403); return res.end(); }
    let body = '';
    req.setEncoding('utf8');
    req.on('data', c => { body += c; if (body.length > 5e6) req.destroy(); });
    req.on('end', () => fs.writeFile(path.join(ROOT, url), body, err => { res.writeHead(err ? 500 : 204); res.end(); }));
    return;
  }
  const file = path.join(ROOT, url === '/' ? 'index.html' : url);
  if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(PORT, () => console.log(`Torch Canvas running at http://localhost:${PORT}`));
