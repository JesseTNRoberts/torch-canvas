<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg">
    <img src="assets/wordmark-light.svg" alt="Torch Canvas" width="440">
  </picture>
</p>

# Torch Canvas

A visual editor for neural networks that writes PyTorch as you draw, and turns PyTorch modules back into diagrams.

## Run

    npm start          # serves http://localhost:5173 — open it in Chrome

It opens with the residual MLP example; more examples are under Open / import. Selecting anything opens a details panel on the right of the canvas. **New model** starts an editable Input → Linear → ReLU → Linear → Output network; add layers by clicking or dragging them from the Layers list. No dependencies. `index.html` is self-contained; `npm run build` regenerates it after editing `src.html` (UI), `core.js` (graph model, shape inference, code generation, importers), `engine.js` (variance prediction), `export.js` (train.py) or anything in `examples/`. `npm test` checks the diagram → code → diagram round trips (including losses) and the exports.

## Tabs

Every model opens in its own tab: New model, an example, a file you open and pasted code each get a tab, and opening a block adds a tab for it next to its model. Each model keeps its own undo history and file. × closes a tab (closing the last one leaves a fresh model). Open models are restored when you come back.

## Library: example networks and blocks

Open / import lists example networks and blocks, each opening in a new tab and leaving the current model alone.

**Every block is a library block, and library blocks are shared.** Each has one definition, and every network that uses it links to it: blocks added from Components → Blocks → Library, blocks you make (New block, Save selection as block), the blocks in the example networks (GPT2Block, BertLayer and LlamaDecoderLayer ship in `examples/blocks/` too), and the blocks of any file you open. Edit the block anywhere, on its own (opened from Open / import, where its tab and code show just the block) or inside any network that uses it, and every open network updates; networks opened later get it too. With `npm start` the change is written back to its file in `examples/blocks/` (a `.py` block is rewritten in the app's code style); without the server it is kept in this browser. Undo inside the block's tab undoes the shared edit; undoing other steps in a network leaves the block alone. There are no unlinked copies. A block's sizes are only defaults, so a network that needs other dimensions sets them on its use; for a block that should actually differ, right-click it in Components → Blocks → **Save as a new library block**, which makes a new shared block (a new file in `examples/blocks/` with `npm start`).

To add your own, drop a file into `examples/networks/` or `examples/blocks/`:

- a `.py` file with an `nn.Module` (for a block, the module itself becomes the block; its constructor defaults become the block's default sizes), with optional comment lines at the top:

      # title: MLP, 10 layers, ReLU
      # description: A plain deep MLP …
      # labels: h=hidden            (blocks: display names for sizes)

- or a `.json` diagram saved with Copy graph JSON, wrapped as `{ "title": …, "description": …, "graph": … }`.

A network file whose name starts with `_` (e.g. `_scratch.json`) is ignored: it stays in `examples/networks/` but doesn't appear under Open / import.

With `npm start` new files show up right away (the server reads the folder on each request). `npm run build` bakes them into `index.html` for the single-file version. Included: Residual MLP, Residual MLP with a block, MLP 10 layers ReLU, MLP 10 layers Tanh, Pre-norm residual MLP, Decoder-only transformer (fully expanded), GPT-2 (124M, 12 × GPT2Block, tied LM head), BERT-base (masked LM, 12 × BertLayer), Llama 2 7B (32 × LlamaDecoderLayer), Qwen2.5-VL 3B (same output as transformers); blocks GatedBlock, PreNormMLP, SwiGLU, LinearReLU, GPT2Block, BertLayer, LlamaDecoderLayer, Qwen2_5_VLVisionBlock, Qwen2_5_VLVisionStage, Qwen2_5_VLPatchMerger, Qwen2_5_VLDecoderLayer.

## Components

The left panel has a tab for each kind of component, grouped by what they are in PyTorch: **Inputs & outputs** (Input arrow, Output arrow), **Layers** with learned weights (Linear, Embedding, Position embedding), **Norms** (LayerNorm, RMSNorm, BatchNorm, GroupNorm, InstanceNorm), **Operations** without weights (Activations; Math: add, multiply, subtract, divide, matmul; Attention: split heads, merge heads, transpose, scale 1/√d, causal mask; Tensor: concat, dropout, custom op), and **Blocks**, your own. Search looks across all tabs. Hover a component for what it does and its PyTorch name; right-click for more (blocks: open, rename, save as a new library block, delete).

### Attention and the transformer example

Open / import ▸ "Decoder-only transformer, fully expanded" is a 2-layer GPT-style decoder with nothing hidden in a block: token + position embeddings, then per layer LayerNorm → q, k, v Linears → split heads → q scaled by 1/√d, q @ kᵀ → causal mask → softmax → @ v → merge heads → output Linear + residual, then LayerNorm → Linear → GELU → Linear + residual; a final LayerNorm and the LM head (vocab 50257, context 1024, d 768, 12 heads, ff 3072 — 92.2M parameters). Click an embedding's size to change d everywhere it must match, or the head count on a split-heads box; a head count that doesn't divide d turns it red. The attention ops write small helpers into the code (`split_heads`, `merge_heads`, `causal_mask`), and code that uses them imports back into the same diagram.

Open / import ▸ "GPT-2 (124M architecture)" is the full GPT-2 small built from one block: token and position embeddings → dropout → **GPT2Block × 12** → final LayerNorm → LM head. Double-click the block to open it: LayerNorm → q, k, v → 12 heads → causal attention with dropout → output projection + residual, then LayerNorm → Linear → GELU (tanh approximation) → Linear → dropout + residual. It uses GPT-2's initialization: N(0, 0.02) for every Linear and both embeddings, zero biases, and the two residual projections scaled by 1/√(2·12). Checked against an independent PyTorch GPT-2 (nanoGPT-style, fused `c_attn`): same outputs to 5·10⁻⁶ after copying the weights across. The LM head shares its weights with the token embedding, as in GPT-2, so it has GPT-2's 124,439,808 parameters. The one difference from the original: q, k and v are three Linears rather than one fused `c_attn` (the same model).

## Losses

Components → **Loss** has cross-entropy, binary cross-entropy (with logits), mean squared error, L1 and Huber. A loss is wired for you when you add it: its first input is the model's output and its second is a target. A language model's loss uses its own token input (next-token prediction); otherwise a target Input arrow is added, shaped for the loss (class indices for cross-entropy, the output's shape for the rest). An Input that only feeds losses is a **target**: it's marked "· target" and isn't an argument of `forward()`.

Cross-entropy's **Targets** setting picks class indices from an input, the **next token** (the output at position t is scored on input token t + 1, as in GPT-2 and Llama), or **masked tokens** (BERT: train.py masks a share of the tokens, 80% → the mask id, 10% → a random token, 10% unchanged, and only those are scored). Several losses add up, each times its weight.

In the code, the model gets three methods:

- `forward()` returns the raw outputs (logits), which is what PyTorch's losses expect;
- `loss(outputs…, targets…)` is the loss components, e.g. `F.cross_entropy(logits[:, :-1].flatten(0, -2), idx[:, 1:].flatten())`;
- `predict()` is `forward()` plus the final activation the loss leaves out: softmax for cross-entropy, sigmoid for binary cross-entropy, nothing for regression.

A loss on an output that already goes through softmax (or sigmoid for BCE) gets a warning: remove it, `predict()` applies it. Code with these `loss()` methods imports back into the same diagram. GPT-2, Llama 2 and the decoder example come with next-token cross-entropy, BERT with masked-token cross-entropy.

## Export for training

**Export…** writes two files: `model.py` (the code panel's code) and `train.py`, a complete training script. The dialog sets the optimizer (SGD, Adam, AdamW, RMSprop, with their settings and weight decay, by default not on biases, norms and embeddings), the learning-rate schedule (constant, cosine, linear, step or one-cycle, with warmup), batch size, sequence length, epochs or steps, gradient accumulation, gradient clipping, precision (fp32, or bf16 / fp16 mixed precision), device, seed, logging, validation and checkpoint intervals, and, under Advanced, `torch.compile`, gradient checkpointing of the model's blocks, an EMA of the weights, and resuming from a checkpoint. train.py is written from those choices: the values sit in a block of constants at the top (`BATCH_SIZE = 32`, `LR = 3e-4`, …), and only what you picked appears in the code: just your optimizer's constructor, just your schedule (PyTorch's own `LinearLR`, `CosineAnnealingLR`, `StepLR`, `OneCycleLR`, joined by `SequentialLR` for warmup), autocast only when mixed precision is on, and accumulation, clipping, EMA, gradient checkpointing, `torch.compile`, resuming and Weights & Biases logging only when turned on. Nothing is switched at run time: a script for fp32 without W&B contains no autocast and no wandb code at all, and an epoch-based run is a plain `for epoch in range(EPOCHS)` loop.

train.py runs as is on stand-in random data with the right shapes and dtypes (token ids for token inputs, class indices for cross-entropy targets); replace `make_dataset()` with your data. It checkpoints to `--out-dir` and resumes with `--resume`. Weights & Biases logging is built in behind `--wandb` (off unless you tick it; `pip install wandb`). Download the two files, or in Chrome save both into a folder. The settings are saved with the model.

Checked by running the exported scripts in PyTorch for every loss kind and for GPT-2, BERT (full size) and a reduced Llama, with SGD, Adam, AdamW and RMSprop, every schedule, gradient accumulation and clipping, EMA, bf16, gradient checkpointing, `torch.compile`, and resuming from a checkpoint.

## Plot variance

**Plot variance** (next to Fit, under the canvas) shows how the variance of the activations and of the gradients changes from Linear to Linear right after initialization, on a log scale, from input to output:

- **forward**: variance of each Linear's output (or input) for a standard Gaussian input;
- **backward**: variance of the gradient at each Linear's output (or input) for a standard Gaussian gradient at the model's output.

Nothing is run. The plot is worked out from the variance rules of each layer (`engine.js`): weights are averaged out, so a Linear's output is Gaussian with Var = fan_in · Var(w) · E[x²] + Var(b), where Var(w) follows its init (PyTorch default 1/(3·fan_in), Kaiming 2/fan_in, Xavier 2g²/(fan_in + fan_out), normal σ², …). Activations, gates, residual adds and norms are applied to sampled values of one feature, so correlations such as σ(w)·w come out exactly. For sequence models each feature is followed across positions, so what positions share (biases, attention averages) is kept, and attention runs on those values with simulated heads. Change an init (right-click a Linear or an Embedding) and press **Replot** to compare. Hover a point to highlight its layer; click to select it. For sequence models the plot has a box to try other lengths (up to 128 positions are simulated).

Checked against real PyTorch (several fresh inits, hooks on every `nn.Linear`): within a few percent for the MLP examples and across every init, norm and activation; for the transformers (GPT-2, and BERT and Llama at reduced width) forward values are within about 10% and gradients mostly within a few percent, with the largest gaps, up to about 35%, on the query and key gradients of deeper layers. Big models are worked out in a background worker so the page stays responsive (Llama 2 7B takes a few seconds).

## Editing

Select an element and press Delete (or Cmd/Ctrl+Delete) to remove it. Every network starts at an Input arrow and ends at an Output arrow; place them from Components → Layers.

## Sizes

Every Linear layer shows its actual size, written like PyTorch's weight shape (`512 × 256` = 256 in, 512 out). Click that size on the layer to change it. Input sizes, and norm widths, follow whatever flows in. Branch sizes (`e × 512`, where `e` is the batch) appear when you hover or click a branch or select a layer; "Sizes: always" in the canvas corner shows them all.

Layers whose outputs must match (both inputs of ⊙ or +) are linked, so changing one changes its partners; choose "Only this layer" to size one on its own. Any operation whose inputs then disagree turns red with a “!” and a note such as `512 ≠ 768`. This covers elementwise ops (⊙, +, −, ÷, with broadcasting), matrix multiply (`@`, `torch.matmul`: the left input's last size must equal the right input's second-to-last), and concatenation. Selecting the operation offers one-click fixes.

The generated code uses actual numbers (`nn.Linear(256, 512)`). Tick "named sizes" in the code panel for constructor arguments instead (`def __init__(self, d_in=784, f=256, …)`). Opening a file follows that file's style.

## Shaping connections

Connections can be styled without touching the code:

- **Move one:** drag any segment of a connection; it stays square-cornered, and a segment that meets a layer gets a short jog so the line still leaves and enters the layer the same way. The bends are saved with the network; the ends follow the layers when you move them. Double-click a connection (or right-click → Reset route) to put it back on the automatic path. Auto layout resets all routes.
- **Colour, weight, dash:** click a connection (Shift-click for several) and set its colour (a palette or any colour), weight (hairline to heavy) and line (solid, dashed, dotted) in the details panel; the arrowhead takes the colour. Right-click → Reset look goes back to the automatic style.

Both are only the drawing: the generated code is the same, and SVG export prints them as shown.

## Exporting a figure

**Export SVG** (below the canvas) saves the diagram exactly as it is shown, for papers and slides: the current tab, in the light theme on a white background, without the editing extras (connection dots, Open buttons, selection), with every style written into the file. Size labels come out as on the canvas, so set **Sizes: always** to include all of them. Drag any size label to move it where it reads best (a click still opens its size); double-click it to put it back. Label positions are saved with the network. The text uses the app's fonts (IBM Plex Sans and Mono, falling back to system fonts), so install them, or convert text to outlines in a vector editor, for exact typography elsewhere.

## Custom blocks

A block is a reusable piece of network that becomes its own class in the code. Each use of it in the diagram is one node that creates and calls that class (`self.block1 = GatedBlock(256, 512)` and `x = self.block1(x)`). Every block is a library block (see Library above): one definition shared by every network that uses it, and every use of a block shares one class in the code.

- **Make one:** select layers (Shift-click, or Shift-drag a box) and choose "Save selection as block" (Components panel or right-click), or click "New block" to start from an empty one. The selection can be anything, even separate pieces (they just make more inputs and outputs), with one exception: data can't leave it and come back in, because the block would then need its own output to compute an input. The app names the layer to add in that case. Each tensor flowing into the selection becomes an input, and each layer whose result leaves it becomes an output, so a block can have several of each (Input and Output arrows inside the block; their names become the `forward()` arguments).
- **Several outputs:** a block with more than one output has one dot per output along its top edge; drag from a dot to connect that output (hover it for its name and shape). In the code the class returns a tuple and the use unpacks it: `y, h = self.cell(x, h)`. Opening code whose modules return tuples reads them back as blocks with several outputs.
- **Use one:** click or drag it from Components → Blocks. A use takes its input size from the stream. The sizes inside a block are its defaults (they become the class's default arguments, `def __init__(self, d=256, h=512)`); every use starts with them and can set its own, so `block2` can be 1024 wide while the others stay at 512. "Reset to the block's defaults" in the details panel undoes that.
- **Edit one:** press **Open** on a block (or double-click it, or press Edit in the list) to open its inside. A block is a network like any other: a green Input arrow where data comes in, a blue Output arrow where it leaves. Add more Input arrows for more inputs. Each opened block gets its own tab above the canvas next to the model's tab, so you can switch between the model and any number of blocks; each tab keeps its own zoom and position. Changes apply to every use. Esc returns to the model tab; × closes a block tab. Closing a network leaves its open block tabs open, as the library blocks' own tabs (their undo history stays with the closed network).
- **Repeat:** a use can be repeated × k (details panel or right-click) when the block has one input and one output of the same size. It's drawn as copies tiled forward along the flow, marked `× 3` and written as `nn.ModuleList([GatedBlock(256, 512) for _ in range(3)])` with a loop; that pattern reads back as a repeated block on import. "Split into separate uses" turns it into k blocks that can each have their own sizes.
- **Undo a block:** right-click a use and choose "Expand into layers".
- Blocks that aren't used yet still appear as classes in the code. Opening a model whose code is split into classes turns those classes into blocks (each linked to the library: the library block of that name if it is the same block, otherwise a new library block, renamed if a different block already has the name).
- Open / import ▸ "Residual MLP with a block" shows the reference network built from one `GatedBlock` repeated three times.

- **Blocks inside blocks:** a block's tab has the same Components → Blocks list; place a block inside another, or select layers inside a block and save them as a block (it is nested there). Blocks in a selection nest too. A block can't contain itself, directly or through others (those rows are greyed out). In the code each block is its own class and calls the classes of the blocks inside it (`self.mix = Mix(d)`), the inner classes written first; sizes pass through (`Encoder(64, 256)` gives its `Gate` the 256). Code whose modules call other modules opens as nested blocks.
- **Delete one:** right-click it (in this network's list or the Library list) → Delete block. It is deleted from the library and from every network: each use is removed and the connections around it are left open, as if its layers had been deleted; blocks that contain it lose that use. If it is used anywhere (open networks, other blocks, saved example networks), the app lists where and asks first. With `npm start` its file is removed and the saved example networks that used it are rewritten; otherwise they drop it when opened.

### Qwen2.5-VL 3B

Open / import ▸ **"Qwen2.5-VL 3B"** is `Qwen2_5_VLForConditionalGeneration` at the size of Qwen2.5-VL-3B-Instruct, and gives **the same output as the reference implementation in Hugging Face `transformers`**. Its inputs are what Qwen's processor produces: `pixel_values` (the 14 × 14-pixel, 2-frame patches of all images in one list, 1176 values each), `image_grid_thw` (frames, rows, columns of patches per image) and `input_ids` (text with image placeholder tokens).

- **Vision encoder:** patch embedding (a Linear, equal to Qwen's stride-14 Conv3d) → the patches put in window order → **Qwen2_5_VLVisionStage × 4**, each 7 blocks attending within 112-pixel windows and 1 attending over the whole frame (Qwen's 32 blocks, full attention in blocks 8, 16, 24, 32) → **Qwen2_5_VLPatchMerger** (RMSNorm, 2 × 2 patches side by side, Linear → GELU → Linear to 2048) → original order restored. **Qwen2_5_VLVisionBlock**: RMSNorm, attention with bias on q/k/v/proj, 2D rotary positions (each patch's row and column), the attention pattern as an input; SwiGLU 3420 with bias.
- **Language model:** the image features replace the placeholder tokens; **Qwen2_5_VLDecoderLayer × 36** with M-RoPE (time, row, column positions worked out from the ids and the grid; sections 16/24/24, base 1e6), 16 query heads sharing 2 key/value heads, causal; final RMSNorm and an LM head tied to the embedding.
- **3,754,622,976 parameters.** `tools/qwen2_5_vl_weights.py` turns a transformers state dict (e.g. the released checkpoint) into this model's (`hf_to_canvas`; q, k, v come from Qwen's single `qkv`), and, run as a script on the exported code of a tiny copy, builds the reference at that size with random weights and compares logits on three images of different sizes in two sequences: the largest difference is about 6e-6 (float32 rounding), with identical predictions. All 3B weights map onto the model with matching shapes.

The pieces are general components, in Components → Ops: **Attention mask** (the allowed pattern as input 2), **Rotary at positions** (axial: one axis per column of input 2) and **M-RoPE** (sections), and under **Vision–language**: window index, window order and restore, patch positions, window mask, frame mask, place image tokens, M-RoPE positions. Each is a small helper function written above the classes. A block repeated × k can have more inputs than one: the first chains from copy to copy, the others go to every copy (`for blk in self.layers: x = blk(x, pos)`). An input can carry an **Example** for the test at the bottom of the code (an image grid, ids with image tokens); then each input gets a line of its own there, with its shape and sizes in a comment that reads back exactly.

### BERT and Llama 2

Open / import ▸ **"BERT-base (masked LM)"** is `BertForMaskedLM`: word, position and token-type embeddings (two inputs: `input_ids`, `token_type_ids`) → LayerNorm → dropout → **BertLayer × 12** → the masked-LM head (Linear → GELU → LayerNorm → decoder). Inside the block attention is bidirectional (no mask) and the block is post-norm: add, then LayerNorm (eps 1e-12) after attention and after the feed-forward. The decoder shares its weights with the word embedding. 109,514,298 parameters, initialization N(0, 0.02) with zero biases.

Open / import ▸ **"Llama 2 7B"** is `LlamaForCausalLM`: token embedding → **LlamaDecoderLayer × 32** → RMSNorm → LM head (not tied). Inside the block: RMSNorm → q, k, v → 32 heads of 128 → **rotary position embeddings** on q and k → causal attention → output projection + residual; RMSNorm → SwiGLU MLP, down(silu(gate(x)) · up(x)) + residual. No biases. 6,738,415,616 parameters, initialization N(0, 0.02). The code builds the full model (about 27 GB in float32).

Both were checked against Hugging Face `transformers`: with the same code at a small size and the weights copied across, `BertForMaskedLM` agrees to 4·10⁻⁷ and `LlamaForCausalLM` exactly; parameter counts match at full size.

**Rotary (RoPE)** is in Components → Ops → Attention. It rotates each pair of features (i, i + D/2) of q or k by an angle that grows with position (base 10000, settable), so q · k depends on the distance between tokens; the code gets a small `apply_rope` helper in the same convention as Hugging Face's Llama.

## Tied weights

Two layers whose weight matrices are the same size can share one weight: a Linear's out × in, or an Embedding's num × dim (for example a language model's head and its token embedding, both vocab × d). Right-click a Linear or an Embedding and choose **Tie to …**, or pick it under **Weights** in the details panel. The choices are the same-size layers in the same model or block, listed by nickname (give a layer a nickname to name it there; layers without one show their code name).

- The code shares the Parameter after both layers exist: `self.lm_head.weight = self.wte.weight`. Importing code with that line ties the layers again.
- The shared weight is counted once in the parameter total, and its init is the init of the layer it's tied to (the tied layer's own init is hidden). The variance plot uses that init too.
- On the canvas a dashed line joins the two layers and the tied one reads "tied to …". If sizes change so the two no longer match, the tied layer turns red.
- Ties stay within one module: inside a block, every use ties its own pair; a tie that would cross into or out of a block is dropped when making or expanding the block. Untie from the same menu.

## Normalization and nicknames

Norm layers are a specific type: LayerNorm, RMSNorm, BatchNorm, GroupNorm or InstanceNorm. Pick one from the Layers list, or switch type by right-clicking a norm or in the inspector (GroupNorm also takes a group count, and every type takes an optional eps). BatchNorm, GroupNorm and InstanceNorm expect channels first, so for sequence inputs (e × t × d) the code transposes around the call, and that idiom is read back as a single norm on import. A GroupNorm whose width isn't divisible by its group count, or an InstanceNorm without a sequence dimension, is marked as a conflict.

Every Linear layer displays as just "Linear". Any element can be given an optional nickname (right-click, or the inspector), shown as a caption beside it; nicknames don't change the code.

## Weight init

Right-click a Linear layer (or use the inspector) to pick its weight init: PyTorch default, Xavier uniform/normal, Kaiming uniform/normal, orthogonal, normal, truncated normal, or zeros, with std or gain where it applies. In a model opened from a file that reuses a class, it applies to the same layer in every copy of that class unless you untick that option. The choice is written as `nn.init` calls right after the layer in `__init__`, and `nn.init` calls are read back on import.

## Open and save files

Use the Open / import menu (Open file, Ctrl+O; Open folder), or drop a `.py` file or folder on the canvas. A folder opens as a project list of every file that defines an `nn.Module`; click a file, or one of its class names to choose the top-level model. Ctrl+S saves the diagram back into the same file: only the model's classes are rewritten, and everything else in the file (imports, training loop, constants) stays as it was. Missing imports are added. If the file changed on disk since it was opened you're asked first, and the save toast has an Undo. While the diagram has no unsaved edits, changes you make to the file in another editor show up in the diagram automatically. Chrome remembers the last folder, and each file keeps its own diagram layout.

This uses Chrome's File System Access API, so it works in the local version (`npm start`, served from localhost). Inside the hosted claude.ai page, Chrome blocks file pickers in the sandboxed frame, so there it opens files read-only and Save copies the code.

## Import

Paste an `nn.Module` anywhere on the page (Ctrl+V), or use Open / import → Paste code. The reader follows `nn.Linear`, `nn.LayerNorm`/`nn.RMSNorm`, activations, `nn.Dropout`, `nn.Sequential`, `nn.ModuleList` (including list comprehensions), `for` loops over module lists, custom submodules, and elementwise math. Other calls are kept as custom ops so the code still regenerates.

For models the reader can't follow, run `export_graph.py` (torch.fx) next to your model and paste the JSON it writes.

## Publishing on GitHub Pages

`index.html` is the whole app, so it runs from any static host.

1. Put this folder in a GitHub repository (`index.html` at the top level) and push it to `main`.
2. In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. The workflow in `.github/workflows/pages.yml` runs on every push to `main`: it runs `npm test`, rebuilds `index.html` (so new files in `examples/` are included) and publishes it. The address appears in the workflow run and under Settings → Pages: `https://<user>.github.io/<repository>/`.

Without Actions: choose **Source: Deploy from a branch** (main, / root) instead; then run `npm run build` and commit `index.html` yourself after changing anything.

On Pages the app works as the single-file version: the example library is what was built into `index.html`, and edits to library blocks, new blocks and open tabs are kept in each visitor's browser. Writing blocks back to `examples/` needs `npm start` on your own machine. Opening and saving files and folders works in Chrome and Edge (Pages is served over https).

## Starting a new repository

    git init -b main
    git add .
    git commit -m "Torch Canvas"
    git remote add origin https://github.com/<user>/torch-canvas.git
    git push -u origin main

Then turn on Pages as described above. The logo is in `assets/`: `logo.svg` (the square mark, also the favicon), and `wordmark-light.svg` / `wordmark-dark.svg` (mark plus name).

## License

MIT, see `LICENSE`.
