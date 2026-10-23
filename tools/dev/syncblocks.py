# Copies library-block layouts (examples/blocks) into the block copies embedded in examples/networks.
# usage (from the repo root): python tools/dev/syncblocks.py
import json, glob, os
lib = {}
for f in glob.glob('examples/blocks/*.json'):
    j = json.load(open(f)); g = j['graph']
    lib['blocks/' + os.path.basename(f)] = (j.get('title'), {n['id']: (n['x'], n['y']) for n in g['nodes']}, {e['id']: e.get('route') for e in g['edges']})
byTitle = {v[0]: k for k, v in lib.items()}
def walk(blocks, out):
    for b in (blocks or {}).values():
        if 'graph' not in b: continue
        key = b.get('lib') or byTitle.get(b.get('name'))
        if key in lib:
            _, pos, rt = lib[key]; k = 0
            for n in b['graph']['nodes']:
                if n['id'] in pos: n['x'], n['y'] = pos[n['id']]; k += 1
            for e in b['graph']['edges']:
                e.pop('route', None)
                if rt.get(e['id']): e['route'] = rt[e['id']]
            out.append(f"{key.split('/')[-1]} {k}/{len(b['graph']['nodes'])}")
        walk(b['graph'].get('blocks'), out)
for f in sorted(glob.glob('examples/networks/*.json')):
    j = json.load(open(f)); out = []; walk(j['graph'].get('blocks'), out)
    if out: json.dump(j, open(f, 'w'), indent=1, allow_nan=False); print(os.path.basename(f), out)
