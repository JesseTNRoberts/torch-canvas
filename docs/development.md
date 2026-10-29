# Development and publishing

Running locally, building, testing, publishing on GitHub Pages and linking to examples.

[← Back to the README](../README.md)

## Run

    npm start          # serves http://localhost:5173 — open it in Chrome

It opens with the residual MLP example; more examples are under Open / import. Double-click a layer or connection (or select it and press Enter) to open its details panel on the right of the canvas. **New model** starts an editable Input → Linear → ReLU → Linear → Output network; add layers by clicking or dragging them from the Layers list. No dependencies. `index.html` is self-contained; `npm run build` regenerates it after editing `src.html` (UI), `core.js` (graph model, shape inference, code generation, importers), `engine.js` (variance prediction), `export.js` (train.py) or anything in `examples/`. `npm test` checks the diagram → code → diagram round trips (including losses), the exports, code edits, torch.fx imports and damaged code (see Checks).

## Checks

Three sets of checks, all run by `.github/workflows/checks.yml` on every pull request, on every push to `main`, and every Monday (so a new PyTorch or browser release that breaks something shows up even when nothing here changed):

- **`npm test`** (Node only, about a minute): every example's code reads back to the same diagram in both code styles; losses, `train.py` and `model.py` exports; code edits keep the layout, nicknames and notes; saved `export_graph.py` output in `test_fixtures/fx/` reads back with the right parameter counts; and 400 pieces of damaged example code (lines lost, text cut short, junk typed in) are each refused with a message rather than a crash or a hang. CI also checks that `index.html` was rebuilt from the sources it contains, and runs a longer fuzz (`npm run fuzz -- 3000`).
- **PyTorch** (`npm run check:pytorch`, needs `torch`): `tools/export_examples.js` writes every example's `model.py` (both styles) and a tiny `train.py`, and `tools/check_examples.py` runs them: the parameter count must be the app's and the output shapes the ones in the code's comments. Models over 40M parameters run on PyTorch's meta device (shapes without memory). With `--train`, the small models train for 2 steps, resume from the checkpoint, and train once more in bf16 with SGD, a one-cycle schedule, EMA, accumulation and clipping.
- **Browser** (`npm run check:browser`, needs `playwright`): `tools/check_browser.py` opens the built app in headless Chromium, opens every example and block, draws it, writes its code and plots its variance; starts the app on damaged saved data; checks that text from pasted code is never run as HTML; and fills the browser's storage to check editing still works and the person is told their tabs aren't being kept. Qwen2.5-VL's variance plot is a known gap (listed in `VARIANCE_UNSUPPORTED`).

The app protects itself from what it saved in the browser: every saved tab and library edit is checked and repaired (`NN.sanitizeGraph`) before it is drawn, and if a start stops part-way anyway, the next start sets the saved data aside under `torchcanvas.backup.<time>.*` and starts fresh. Example tabs with no edits are saved as a reference to the example, which keeps the saved state small.

## Publishing on GitHub Pages

`index.html` is the whole app, so it runs from any static host.

1. Put this folder in a GitHub repository (`index.html` at the top level) and push it to `main`.
2. In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. The workflow in `.github/workflows/pages.yml` runs on every push to `main`: it runs `npm test`, rebuilds `index.html` (so new files in `examples/` are included) and publishes it. The address appears in the workflow run and under Settings → Pages: `https://<user>.github.io/<repository>/`.

Without Actions: choose **Source: Deploy from a branch** (main, / root) instead; then run `npm run build` and commit `index.html` yourself after changing anything.

On Pages the app works as the single-file version: the example library is what was built into `index.html`, and edits to library blocks, new blocks and open tabs are kept in each visitor's browser. Writing blocks back to `examples/` needs `npm start` on your own machine. Opening and saving files and folders works in Chrome and Edge (Pages is served over https).

## Developer tools

`tools/dev/` holds the scripts used while changing the app:

- `node tools/dev/regress.js out/` writes the generated code of every example network (both styles) to `out/`. Run it before and after a change and `diff -r` the two folders: any difference is a change in the code users get.
- `node tools/dev/vcmp.js old/examples examples` compares the variance plot of every example network with a saved copy of `examples/`.
- `python tools/dev/applylayout.py networks/NN_name.json blocks/name.json` lays out residual networks the house way: the stream across the top through the + nodes, each sublayer a column below its +, input bottom-left (needs playwright and a built `index.html`).
- `python tools/dev/syncblocks.py` copies the layouts of `examples/blocks/` into the block copies embedded in the example networks.

The `tools/check_*.py` scripts compare an example's exported code with the reference implementation in PyTorch, and `tools/measure_variance.py` measures what the variance plot predicts.

## Example categories

Every file in `examples/` names its menu category: a `"category"` field next to `"description"` in a `.json`, or a `# category: …` comment line at the top of a `.py`. Network categories are listed in `NET_CATS` and block categories in `BLOCK_CATS` in `src.html`; that order is the menu order. A category that isn't in those lists is shown before **Other**, and a file with no category goes under **Other**.

## Linking to an example

Add `?open=` and an example's name to the address to open that model directly:

    https://<user>.github.io/torch-canvas/?open=gpt2
    https://<user>.github.io/torch-canvas/?open=qwen2_5_vl_3b
    https://<user>.github.io/torch-canvas/?open=swiglu          (a library block)

The name is the file name in `examples/` without the number prefix and extension (`07_gpt2.json` → `gpt2`); the full file name works too. The easiest way to get the link is **Open / import → Copy link to this example** while the example's tab is open. Someone opening the link for the first time sees only that model. If they already have it open, the link switches to that tab. Links point at the examples as published, not at your own edits. To share an edited network, save it into `examples/networks/` and push.

## Starting a new repository

    git init -b main
    git add .
    git commit -m "Torch Canvas"
    git remote add origin https://github.com/<user>/torch-canvas.git
    git push -u origin main

Then turn on Pages as described above. The logo is in `assets/`: `logo.svg` (the square mark, also the favicon), and `wordmark-light.svg` / `wordmark-dark.svg` (mark plus name).
