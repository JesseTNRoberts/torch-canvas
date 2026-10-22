// Builds the single-file app: src.html + core.js + engine.js + export.js + the examples library → index.html
const fs = require('fs');
const path = require('path');
const { scanLibrary } = require('./library.js');
const lib = scanLibrary();
const src = fs.readFileSync(path.join(__dirname, 'src.html'), 'utf8');
const core = fs.readFileSync(path.join(__dirname, 'core.js'), 'utf8');
const engine = fs.readFileSync(path.join(__dirname, 'engine.js'), 'utf8');
const exporter = fs.readFileSync(path.join(__dirname, 'export.js'), 'utf8');
const libJs = 'window.TC_LIBRARY = ' + JSON.stringify(lib).replace(/<\//g, '<\\/') + ';';
fs.writeFileSync(path.join(__dirname, 'index.html'), src.replace('/*CORE*/', () => core).replace('/*ENGINE*/', () => engine).replace('/*EXPORT*/', () => exporter).replace('/*LIBRARY*/', () => libJs));
console.log(`index.html built with ${lib.filter(e => e.kind === 'network').length} networks and ${lib.filter(e => e.kind === 'block').length} blocks`);
