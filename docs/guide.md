# Torch Canvas guide

How to use the editor: the panels, components, editing, sizes, blocks, ties and files.

[← Back to the README](../README.md)

## Tabs

Every model opens in its own tab: New model, an example, a file you open and pasted code each get a tab, and opening a block adds a tab for it next to its model. Each model keeps its own undo history and file. × closes a tab (closing the last one leaves a fresh model). Open models are restored when you come back.

## Components

The left panel has a tab for each kind of component, grouped by what they are in PyTorch: **Inputs & outputs** (Input arrow, Output arrow), **Layers** with learned weights (Linear, Embedding, Position embedding), **Images** (Conv2d, BatchNorm2d, max / average / global average pooling, upsample, flatten), **Norms** (LayerNorm, RMSNorm, BatchNorm, GroupNorm, InstanceNorm), **Operations** without weights (Activations; Math: add, multiply, subtract, divide, matmul; Reshape; Attention: causal mask, plus presets for split heads, merge heads, swap last two and the 1/√(head width) multiply; Tensor: concat, dropout, custom op), and **Blocks**, your own. Search looks across all tabs. Hover a component for what it does and its PyTorch name; right-click for more (blocks: open, rename, save as a new library block, delete).

### Reshape: tensor dimension management

Every operation that only rearranges a tensor is one node, **Reshape**, written as a pattern of named axes, einops-style:

```
split heads:     e t (h d)  →  e h t d      h = n_head      x.unflatten(-1, (n_head, -1)).transpose(1, 2)
merge heads:     e h t d    →  e t (h d)                    x.transpose(1, 2).flatten(2)
kᵀ:              e h t d    →  e h d t                      x.transpose(-2, -1)
ViT patches:     e c h w    →  e (h w) c                    x.flatten(2).transpose(-2, -1)
CNN head:        e c h w    →  e (c h w)                    x.flatten(1)
Mamba:           e t d      →  e t d 1                      x.unsqueeze(-1)
```

`(h d)` is one axis made of two, `1` an axis of size 1, `...` any axes not named; **Sizes** gives all but one part of a split (`h = n_head`), and the last is worked out. The node shows the pattern and the actual shape change (`64 × 768 → 12 × 64²`), so split heads, merge heads, flatten and the rest read as what they are: the same values, arranged differently. Its **Nickname** says what the step is for ("split heads"), and its details spell out each split, merge and move. The code is ordinary PyTorch (unflatten, squeeze, transpose or permute, flatten, unsqueeze, in that order), and code written with those methods reads back as Reshape nodes (a chain like `x.unflatten(…).transpose(1, 2)` as one node). Files that use the older Split heads, Merge heads and Transpose nodes still open.

Any node can carry a **Note**: a sentence or two on why it is there, shown in its details under what it computes. The attention examples use one on the multiply by `(d / n_head) ** -0.5`, the constant that keeps the scores q · k from growing with the head width.

### What a function node shows

Function calls are drawn by what they do rather than by their PyTorch name:
- **eˣ** for `torch.exp`.
- **Reshapes** (flatten, unflatten, squeeze, unsqueeze) show their name above the shape change without the batch, e.g. `64 × 7² → 3136`. When that gets long, sizes the two sides share at the front become …, e.g. `… × 1536 → … × 1536 × 1`.
- **Sum over axis** and **Mean over axis** (Ops → Math) are nodes of their own, with the axis picked by name in their details.
- **Concatenation** puts ‖ between the joined shapes: `64×64² ‖ 64×64² → 128×64²`.
- **Helpers** have plain names, e.g. Qwen's `window_restore` is "undo window sort".

The details of every node start with what it computes: a formula for math, a description for operations that only move values around.

### Recurrence: the Delay

A **Delay** (Ops → Sequence) passes on what it received one time step earlier: out[:, t] = in[:, t − 1] on an e × t × … sequence, zeros at the first step. A connection that closes a loop is allowed when the loop runs through a Delay, so a recurrent network is drawn flat: an RNN is `h = tanh(Linear(x) + Linear(Delay(h)))`, with `h` fed back into the Delay. A loop without a Delay is still refused.

In the code each loop through Delays is a `for t in range(…)` over time: whatever the loop reads from outside (like the input projections) is computed once for the whole sequence and sliced at each step, each Delay is a state variable (named in its settings: h, c) that starts as zeros, and what the rest of the model reads from the loop is collected and stacked back into a sequence. A Delay outside any loop is just a shift along time (`delay(x)`). Code written in that style, including the usual hand-written loop (`h = torch.zeros(…)`, `for t in range(x.size(1))`, `outputs.append(…)`, `torch.stack(outputs, dim=1)`), imports back into Delays.

A Delay can be **drawn split**, like one step of a textbook cell (tick *Split in / out* in its settings): the node becomes h<sub>t−1</sub>, where the state comes in, and a second tab h<sub>t</sub>, where the new state leaves, which you can drag anywhere. A faint wire joins the two, above or below the cell. Only the drawing changes. The LSTM and GRU examples are drawn this way, laid out like the usual diagrams: the cell state (LSTM) or hidden state (GRU) runs across the top, the gates sit in a row below it, and x<sub>t</sub> and h<sub>t−1</sub> come in from below.

- **RNN (Elman)**: h = tanh(W_ih·x + W_hh·h_prev). 5,768 parameters.
- **LSTM**: input, forget, cell and output gates, each from x and h_prev; c = f ⊙ c_prev + i ⊙ g, h = o ⊙ tanh(c), with Delays for h and c. 21,512 parameters.
- **GRU**: reset and update gates, candidate n = tanh(W_in·x + r ⊙ (W_hn·h_prev)), h = (1 − z) ⊙ n + z ⊙ h_prev. 16,264 parameters.

Each takes 16 features per step, has 64 hidden units, reads 8 values out of every step with a Linear and is trained by mean squared error against a target sequence. Each gate has its own pair of Linears (x and h_prev); stacked in PyTorch's gate order (i, f, g, o and r, z, n) they are exactly `nn.LSTM`'s and `nn.GRU`'s `weight_ih_l0` / `weight_hh_l0` and biases, and `python tools/check_recurrent.py` copies a built-in layer's weights in and checks the outputs are the same. They start from `nn.Linear`'s default init rather than the built-in layers' uniform ±1/√hidden.

The variance plot looks at a recurrence one step at a time, without running through time: each Delay in a loop gives the step a previous state that is independent of the current input, and gradients stop at the Delay (no backpropagation through time). The state has the distribution it settles to: starting from zeros, the same step is repeated with the states it produces until their variance stops changing (a handful of repeats). A unit's bias, and what its weights make of the input's average, are kept the same at every repeat, since the weights don't change over time; that is what lets an LSTM's cell state build up. Checked against PyTorch on 60-step sequences with the state detached at every step: within about 7% for the RNN, LSTM and GRU.

## Editing

Select an element and press Delete (or Cmd/Ctrl+Delete) to remove it. Every network starts at an Input arrow and ends at an Output arrow; place them from Components → Layers.

## Sizes

Every Linear layer shows its actual size, written like PyTorch's weight shape (`512 × 256` = 256 in, 512 out). Click that size on the layer to change it. Input sizes, and norm widths, follow whatever flows in. Branch sizes (`e × 512`, where `e` is the batch) appear when you hover or click a branch or select a layer; "Sizes: always" in the canvas corner shows them all.

Layers whose outputs must match (both inputs of ⊙ or +) are linked, so changing one changes its partners; choose "Only this layer" to size one on its own. Any operation whose inputs then disagree turns red with a “!” and a note such as `512 ≠ 768`. This covers elementwise ops (⊙, +, −, ÷, with broadcasting), matrix multiply (`@`, `torch.matmul`: the left input's last size must equal the right input's second-to-last), and concatenation. Selecting the operation offers one-click fixes.

The generated code uses actual numbers (`nn.Linear(256, 512)`). Tick "named sizes" in the code panel for constructor arguments instead (`def __init__(self, d_in=784, f=256, …)`). Opening a file follows that file's style.

## Shaping connections

The automatic path avoids other layers and keeps connections apart. A connection that would cross a layer goes around it in a clear lane. Connections from different layers that would share a line are spaced apart; connections from the same layer stay together. A loss that reads the same tensor as an output arrow is drawn coming from that arrow, so the last layer doesn't look like it has two outputs. The connection itself still comes from the layer, and so does the code. `node tools/route_check.js` (needs Playwright) counts any overlaps and lines through layers in every example.

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

## Freezing

Any layer with weights (Linear, Conv2d, Causal conv, Embedding, a norm with weights, a learned token or Parameter) and any block use can be **frozen**: tick **Freeze ❄** in its details, or right-click it (several at once with a selection). A frozen item shows a snowflake on its corner, and the parameter count shows how much is frozen (`86.57M params · ❄ 85.65M`). In `model.py` it is `self.layer.requires_grad_(False)` right after the layer is made (a repeated block: the whole `nn.ModuleList`), and code written that way reads back frozen. `train.py` gives the optimizer only the weights that still train, and prints how many that is. Gradients still flow through a frozen layer to the layers before it; only its own weights stay put, as when fine-tuning a head on a frozen backbone. Freezing doesn't change the variance plot, which looks at the model right after initialization. A tied layer shares its weight with the other one, so freeze both or neither (the code warns otherwise).

## Tied weights

Two layers whose weight matrices are the same size can share one weight: a Linear's out × in, or an Embedding's num × dim (for example a language model's head and its token embedding, both vocab × d). Right-click a Linear or an Embedding and choose **Tie to …**, or pick it under **Weights** in the details panel. The choices are the same-size layers in the same model or block, listed by nickname (give a layer a nickname to name it there; layers without one show their code name).

- The code shares the Parameter after both layers exist: `self.lm_head.weight = self.wte.weight`. Importing code with that line ties the layers again.
- The shared weight is counted once in the parameter total, and its init is the init of the layer it's tied to (the tied layer's own init is hidden). The variance plot uses that init too.
- On the canvas a dashed line joins the two layers and the tied one reads "tied to …". Drag the line sideways to move it (past the layers it attaches to their right sides); double-click it to put it back. If sizes change so the two no longer match, the tied layer turns red.
- Ties stay within one module: inside a block, every use ties its own pair; a tie that would cross into or out of a block is dropped when making or expanding the block. Untie from the same menu.

### Follows (EMA)

The other kind of tie: a layer or a whole block use can **follow** another of the same kind and sizes, in the model (choose it under **Follows (EMA)** in its details, or click the dotted link to change it). A follower doesn't share the weights; it keeps its own copy that trails behind, an exponential moving average, as I-JEPA's, BYOL's and DINO's target encoders do:

- It starts as a copy (`self.target.load_state_dict(self.encoder.state_dict())`) and gets no gradients (`requires_grad_(False)`); in forward() it runs under `with torch.no_grad():`, so nothing flows back through it either.
- `ema_update(progress)` (written on the model) moves each follower's weights toward what it follows, w ← m·w + (1 − m)·w′, with the momentum m going from its first value to its second over training (0.996 → 1.0 by default; equal values: constant). `train.py` calls `model.ema_update((step + 1) / total_steps)` after every optimizer step.
- Code written that way reads back as the same tie, momentum included. The parameter chip counts followers with the frozen weights (`❄`), and the variance plot shows their forward values only.
- On the canvas a dotted blue link joins the two, labeled "follows · EMA".

## Normalization and nicknames

Norm layers are a specific type: LayerNorm, RMSNorm, BatchNorm, GroupNorm or InstanceNorm. Pick one from the Layers list, or switch type by right-clicking a norm or in the inspector (GroupNorm also takes a group count, and every type takes an optional eps). BatchNorm, GroupNorm and InstanceNorm expect channels first, so for sequence inputs (e × t × d) the code transposes around the call, and that idiom is read back as a single norm on import. A GroupNorm whose width isn't divisible by its group count, or an InstanceNorm without a sequence dimension, is marked as a conflict.

Every Linear layer displays as just "Linear". Any element can be given an optional nickname (right-click, or the inspector), shown as a small title inside its box (beside it for the round + ⊙ @ operators); nicknames don't change the code.

## Weight init

Right-click a Linear layer (or use the inspector) to pick its weight init: PyTorch default, Xavier uniform/normal, Kaiming uniform/normal, orthogonal, normal, truncated normal, or zeros, with std or gain where it applies. In a model opened from a file that reuses a class, it applies to the same layer in every copy of that class unless you untick that option. The choice is written as `nn.init` calls right after the layer in `__init__`, and `nn.init` calls are read back on import.

## Open and save files

Use the Open / import menu (Open file, Ctrl+O; Open folder), or drop a `.py` file or folder on the canvas. A folder opens as a project list of every file that defines an `nn.Module`; click a file, or one of its class names to choose the top-level model. Ctrl+S saves the diagram back into the same file: only the model's classes are rewritten, and everything else in the file (imports, training loop, constants) stays as it was. Missing imports are added. If the file changed on disk since it was opened you're asked first, and the save toast has an Undo. While the diagram has no unsaved edits, changes you make to the file in another editor show up in the diagram automatically. Chrome remembers the last folder, and each file keeps its own diagram layout.

This uses Chrome's File System Access API, so it works in the local version (`npm start`, served from localhost). Inside the hosted claude.ai page, Chrome blocks file pickers in the sandboxed frame, so there it opens files read-only and Save copies the code.

## Import

Paste an `nn.Module` anywhere on the page (Ctrl+V), or use Open / import → Paste code. The reader follows `nn.Linear`, `nn.LayerNorm`/`nn.RMSNorm`, activations, `nn.Dropout`, `nn.Sequential`, `nn.ModuleList` (including list comprehensions), `for` loops over module lists, custom submodules, and elementwise math. Other calls are kept as custom ops so the code still regenerates.

For models the reader can't follow, run `export_graph.py` (torch.fx) next to your model and paste the JSON it writes.
