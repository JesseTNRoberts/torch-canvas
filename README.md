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

With `npm start` new files show up right away (the server reads the folder on each request). `npm run build` bakes them into `index.html` for the single-file version. Included: Residual MLP, Residual MLP with a block, MLP 10 layers ReLU, MLP 10 layers Tanh, Pre-norm residual MLP, Decoder-only transformer (fully expanded), GPT-2 (124M, 12 × GPT2Block, tied LM head), BERT-base (masked LM, 12 × BertLayer), Llama 2 7B (32 × LlamaDecoderLayer), Qwen2.5-VL 3B (same output as transformers), Basic CNN (MNIST), ResNet-50 (same output as torchvision), YOLOv3 (same output as Darknet's yolov3.cfg), RNN, LSTM and GRU (drawn with Delays, same output as nn.RNN / nn.LSTM / nn.GRU), ViT-B/16 (same output as torchvision), Mamba-130M (same output as transformers), VAE (MNIST, same as PyTorch's examples/vae), U-Net (same output as milesial/Pytorch-UNet), the original encoder–decoder Transformer (same output as nn.Transformer), I-JEPA (small; same output as facebookresearch/ijepa); blocks GatedBlock, PreNormMLP, SwiGLU, LinearReLU, GPT2Block, BertLayer, LlamaDecoderLayer, Qwen2_5_VLVisionBlock, Qwen2_5_VLVisionStage, Qwen2_5_VLPatchMerger, Qwen2_5_VLDecoderLayer, Bottleneck, BottleneckDown, DarknetConv, DarknetResidual, YoloConvSet, YoloDetect, ViTEncoderBlock, MambaBlock, DoubleConv, Down, Up, EncoderLayer, DecoderLayer, JEPABlock, Encoder.

## Components

The left panel has a tab for each kind of component, grouped by what they are in PyTorch: **Inputs & outputs** (Input arrow, Output arrow), **Layers** with learned weights (Linear, Embedding, Position embedding), **Images** (Conv2d, BatchNorm2d, max / average / global average pooling, upsample, flatten), **Norms** (LayerNorm, RMSNorm, BatchNorm, GroupNorm, InstanceNorm), **Operations** without weights (Activations; Math: add, multiply, subtract, divide, matmul; Attention: split heads, merge heads, transpose, scale 1/√d, causal mask; Tensor: concat, dropout, custom op), and **Blocks**, your own. Search looks across all tabs. Hover a component for what it does and its PyTorch name; right-click for more (blocks: open, rename, save as a new library block, delete).

### Attention and the transformer example

Open / import ▸ "Decoder-only transformer, fully expanded" is a 2-layer GPT-style decoder with nothing hidden in a block: token + position embeddings, then per layer LayerNorm → q, k, v Linears → split heads → q scaled by 1/√d, q @ kᵀ → causal mask → softmax → @ v → merge heads → output Linear + residual, then LayerNorm → Linear → GELU → Linear + residual; a final LayerNorm and the LM head (vocab 50257, context 1024, d 768, 12 heads, ff 3072 — 92.2M parameters). Click an embedding's size to change d everywhere it must match, or the head count on a split-heads box; a head count that doesn't divide d turns it red. The attention ops write small helpers into the code (`split_heads`, `merge_heads`, `causal_mask`), and code that uses them imports back into the same diagram.

Open / import ▸ "GPT-2 (124M architecture)" is the full GPT-2 small built from one block: token and position embeddings → dropout → **GPT2Block × 12** → final LayerNorm → LM head. Double-click the block to open it: LayerNorm → q, k, v → 12 heads → causal attention with dropout → output projection + residual, then LayerNorm → Linear → GELU (tanh approximation) → Linear → dropout + residual. It uses GPT-2's initialization: N(0, 0.02) for every Linear and both embeddings, zero biases, and the two residual projections scaled by 1/√(2·12). Checked against an independent PyTorch GPT-2 (nanoGPT-style, fused `c_attn`): same outputs to 5·10⁻⁶ after copying the weights across. The LM head shares its weights with the token embedding, as in GPT-2, so it has GPT-2's 124,439,808 parameters. The one difference from the original: q, k and v are three Linears rather than one fused `c_attn` (the same model).

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

### Mamba: a selective state space model

**Mamba-130M** (`?open=mamba_130m`) is `state-spaces/mamba-130m` as transformers' `MambaForCausalLM`: token embeddings, **MambaBlock × 24**, a final RMSNorm and an LM head tied to the embeddings, trained on the next token. Each MambaBlock is RMSNorm, then the mixer, then the residual add:

- `in_x` and `in_z` (Mamba's `in_proj`, split in two) take d = 768 to d_inner = 1536;
- x goes through a **Causal conv** (Ops → Sequence: `nn.Conv1d` with one filter of width 4 per channel, seeing only the current and earlier steps, written as `causal_conv(self.conv1d, x)`) and SiLU, giving u;
- `x_dt`, `x_B` and `x_C` (Mamba's `x_proj`, split in three) read Δ's low-rank input (48), B and C (16 each) from u; `dt_proj` and **Softplus** make the time step Δ, one per channel and step;
- the scan is drawn flat with a **Delay**, as in the RNNs: h = exp(Δ·A) ⊙ h_prev + Δ·u·B over a d_inner × d_state state, A = −exp(A_log). Δ·A and Δ·u·B are built with `.unsqueeze(-1)` / `.unsqueeze(-2)` and broadcasting, and the loop itself is just ⊙ and +;
- y = Σ_n h·C (`.sum(-1)`) + D·u, gated by SiLU(z), then `out_proj`.

`A_log` and `D` are **Parameter** nodes (Ops → Sequence: a learned tensor with a shape and an init: zeros, ones, or log 1 … N, Mamba's S4D-real A). `dt_proj`'s bias starts as Mamba's: softplus(bias) log-uniform in [0.001, 0.1] (a Linear's **Bias init**, written `dt_bias_(self.dt_proj.bias)`), and the embeddings start N(0, 0.02²). There are 129,135,360 parameters, the same as the reference. `python tools/check_mamba.py model.py` copies transformers' weights in (splitting `in_proj` and `x_proj`) and compares: one block agrees to float32 rounding, and the 24-block logits agree to 2 × 10⁻⁴ of their size, with the same argmax everywhere (transformers runs the scan in its own order).

The variance plot runs Mamba's scan along the simulated positions, forward and back. Because the loop has no weights of its own, there is nothing to settle and no need to cut it at the Delay. The state's d_state axis is carried along (8 of the 16 entries per simulated channel, sharing its Δ and u), and B and C, which every channel shares, get gradients summed over all of them. Checked against PyTorch (8 × 64 tokens, hooks on every Linear and Conv1d): forward values have a median error of 1%, and gradients a median of 5% (largest about 60%, except x_C, whose 16 outputs make both the plot and the measurement noisy, up to 2× either way).

### Encoder–decoder and generative models

- **Transformer (encoder–decoder)** (`?open=transformer_base`): "Attention Is All You Need", base size (d = 512, 8 heads, feed-forward 2048, 6 + 6 layers, dropout 0.1), computed as `nn.Transformer` does it: post-norm layers, plus a final LayerNorm on the encoder and on the decoder. Source and target tokens share one embedding (37,000 tokens), scaled by √d (written `x * d ** 0.5`) and given fixed sine/cosine positions: a **Sin/cos positions** table (Ops → Sequence, `sinusoidal_pe(x)`, shaped by its input) added with a +, so the code reads `x = x + sinusoidal_pe(x)`. **EncoderLayer × 6** turns the source into the memory. **DecoderLayer × 6** takes two inputs: the target sequence, and the memory, which goes to every copy. Each decoder layer has causal self-attention, then **cross-attention**, where queries come from the decoder and keys and values from the memory (the scores are t × s), then the feed-forward network. The output projection shares the embedding's weights, and the loss is next-token cross-entropy with label smoothing 0.1. It has 63,084,544 parameters, and its init is `nn.Transformer`'s: Xavier uniform, with q, k and v at gain 1/√2 so they match the packed 3d × d projection. Source and target have their own lengths, s and t; the example call shows each with a size comment (`# (e=8, s=64)`), which reads back exactly. `python tools/check_transformer.py model.py` copies `nn.Transformer`'s weights in (splitting `in_proj`) and gets the same logits.
- **U-Net** (`?open=unet`): milesial/Pytorch-UNet (`bilinear=False`), 3 → 2 classes at 256 × 256. A **DoubleConv** ((Conv 3×3 → BatchNorm → ReLU) × 2), four **Down** blocks (max pool, then DoubleConv, a nested block), and four **Up** blocks with two inputs: the deeper map, doubled by a **ConvTranspose2d** (Images; a Conv2d with Transposed ticked), and the skip connection from the way down, joined along the channels before a DoubleConv. A 1×1 convolution gives a score per class per pixel, and the cross-entropy is per pixel against a mask of class indices (`predict()` applies softmax over the channels). It has 31,037,698 parameters, and `python tools/check_unet.py model.py path/to/Pytorch-UNet` gets the same outputs in training and eval mode.
- **VAE (MNIST)** (`?open=vae_mnist`): PyTorch's examples/vae. The encoder 784 → 400 (ReLU) gives μ and log σ² (20 each). z = μ + ε·exp(½·log σ²), with ε from a **Noise** node (Ops → Sequence: `torch.randn_like`, so gradients reach μ and σ: the reparameterization trick). The decoder 20 → 400 → 784 gives pixel logits. forward() returns all three, and the loss is the negative ELBO per example: 784 × binary cross-entropy (the sum over pixels) plus **KL to N(0, 1)** (Losses: its two inputs are the mean and the log-variance, both outputs, written `kl_normal(mu, logvar)`). It has 652,824 parameters, and `python tools/check_vae.py model.py` gets the same reconstruction, and the same loss per example, as examples/vae with the same noise.

The variance plot covers all three: Noise is fresh unit Gaussian values, and a transposed convolution reads one input pixel per output pixel through the kernel offset it sits at (PyTorch's default init takes ConvTranspose2d's fan-in from its output channels). The sin/cos table is a constant added feature by feature, and cross-attention runs like self-attention (both sequences are simulated at the longer length). Checked against PyTorch, forward values are within 13% for the VAE, 6% for U-Net and 10% for the Transformer. Gradients have a median error of 3%, 8% and 9%; the Transformer's largest gap is 43%, on query and key gradients as for the other transformers.

### Self-supervised: I-JEPA

**I-JEPA (small)** (`?open=ijepa_small`): facebookresearch/ijepa's encoder and predictor at 32 × 32 with 4 × 4 patches (64), width 192, 6 encoder and 3 predictor layers (**JEPABlock**, a pre-norm ViT block; **Encoder**, six of them and a LayerNorm). It learns by predicting, in embedding space, what it can't see:

- The **context encoder** gets the patch embeddings plus fixed 2-D sine/cosine positions (a frozen **Parameter**), keeps only the context patches (**Take positions** with the index input `ctx_idx`), and runs the Encoder.
- The **predictor** maps that to width 96, adds the positions of the context patches, appends a learned mask token plus the position of every target patch (`tgt_idx`), runs JEPABlock × 3 and a LayerNorm, keeps the appended positions (**Last positions**) and maps back to 192: `pred`.
- The **target encoder** sees the whole image; its output, normalized without weights (a LayerNorm with Scale and shift off), at the target positions is `targets`. It and its patch embedding **follow** the context encoder's (EMA, 0.996 → 1.0).
- The loss is smooth L1 (Huber, δ = 1) between `pred` and `targets`, two outputs of the model.

`train.py` draws I-JEPA's masks for every batch (`sample_masks`: four target blocks of 15–20% each, aspect 0.75–1.5, and a context block of 85–100% with the targets taken out; I-JEPA predicts each block in a pass of its own, here the four together) and calls `ema_update` after every step. 5,749,344 parameters (3.05M trained; the target encoder follows; I-JEPA also keeps a second copy of the fixed position table, 12,288 more). `python tools/check_ijepa.py model.py path/to/ijepa` copies its weights in and gets the same predictions and targets for the same masks, and the same targets after an EMA step. The variance plot is within 10% of PyTorch going forward and a median of 5% for the gradients (largest 38%, on query gradients).

### Image models

Images flow as e × C × H × W (batch, channels, height, width). **Conv2d** has a kernel, stride and padding; its input channels follow the stream like a Linear's input, and you set the output channels (click the size on the box). Every layer shows how the size changes: ⌊(size + 2·padding − kernel) / stride⌋ + 1 per side, so changing a stride or a pool updates everything after it, including the width of the first Linear after a flatten. **BatchNorm2d** normalizes each channel; pooling (max, average, global average) and **Upsample** (nearest or bilinear) have no weights; concat with `dim=1` joins feature maps. LeakyReLU is under Activations. A block whose input is an image shows its channels and size, e.g. `256 @ 56² → 512 @ 28²`. Code with `nn.Conv2d`, `nn.BatchNorm2d`, `nn.MaxPool2d` / `F.max_pool2d`, `F.adaptive_avg_pool2d`, `F.interpolate`, `nn.LeakyReLU` and `x.flatten(1)` imports back into the same diagram; the image size comes from the test call at the bottom, or, without one, from the size the first Linear after the convolutions expects.

- **Basic CNN (MNIST)**: no blocks. Conv 3×3 → ReLU → max pool, twice (32 and 64 channels), flatten, Linear 3136 → 128 → ReLU → dropout → Linear → 10, with cross-entropy. 421,642 parameters.
- **ResNet-50 (ImageNet)**: torchvision's architecture (v1.5, stride on the 3×3): a 7×7 stem and max pool, then four stages, each a **BottleneckDown** (changes the channels, stride 2 except the first, with a 1×1 convolution on the shortcut) followed by **Bottleneck × 2, 3, 5, 2**; global average pooling and a 1000-way Linear. 25,557,032 parameters, the same as `torchvision.models.resnet50`, and the same outputs after copying its weights across.
- **YOLOv3 (416, COCO)**: Darknet-53 from **DarknetConv** (Conv → BatchNorm → LeakyReLU 0.1) and **DarknetResidual** blocks (× 1, 2, 8, 8, 4), then three detection heads (**YoloConvSet** → **YoloDetect**): 13 × 13, and 26 × 26 and 52 × 52 after a 1×1 convolution, upsampling and joining the backbone's features. Three outputs of 255 = 3 × (4 + 1 + 80) channels per cell (the raw head outputs; decoding boxes and the YOLO loss aren't part of the diagram). 61,949,149 parameters, the same as Darknet's `yolov3.cfg`, and the same outputs as a model built straight from that file.

- **ViT-B/16 (ImageNet)**: torchvision's `vit_b_16`. A 16 × 16 convolution with stride 16 cuts the image into 14 × 14 patches of 768 features; `.flatten(2)` and a transpose make them a sequence of 196; a **Learned token** (the class token, under Operations → Sequence) goes in front, learned position embeddings (197) are added, then **ViTEncoderBlock × 12** (pre-norm: LayerNorm, 12-head self-attention without a mask, residual; LayerNorm, Linear 768 → 3072 → GELU → 768, residual; LayerNorm eps 1e-6), a final LayerNorm, **Select position** 0 (the class token) and a 1000-way Linear. 86,567,656 parameters, the same as torchvision, and the same outputs after copying its weights across (its packed q/k/v projection is split into three Linears).

`python tools/check_image_models.py` (needs torch and torchvision) repeats the ResNet and YOLO comparisons, `python tools/check_vit.py model.py` the ViT one. The variance plot covers them too (see Plot variance).

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

**Plot variance** (next to Fit, under the canvas) shows how the variance of the activations and of the gradients changes from layer to layer (every Linear, Conv2d, ConvTranspose2d and causal Conv1d) right after initialization, on a log scale, from input to output:

- **forward**: variance of each layer's output (or input) for a standard Gaussian input;
- **backward**: variance of the gradient at each layer's output (or input) for a standard Gaussian gradient at the model's output.

Nothing is run. The plot is worked out from the variance rules of each layer (`engine.js`): weights are averaged out, so a Linear's output is Gaussian with Var = fan_in · Var(w) · E[x²] + Var(b), where Var(w) follows its init (PyTorch default 1/(3·fan_in), Kaiming 2/fan_in, Xavier 2g²/(fan_in + fan_out), normal σ², …). Activations, gates, residual adds and norms are applied to sampled values of one feature, so correlations such as σ(w)·w come out exactly. For sequence models each feature is followed across positions, so what positions share (biases, attention averages) is kept, and attention runs on those values with simulated heads. Change an init (right-click a Linear or an Embedding) and press **Replot** to compare. Hover a point to highlight its layer; click to select it. For sequence models the plot has a box to try other lengths (up to 128 positions are simulated).

For image models a Conv2d is a Linear over each k × k window (fan_in = C_in·k²), and the covariance between pixels is carried along: its output at pixels P and Q has covariance C_in · Var(w) · Σ over kernel offsets of E[x x'] at the two input pixels, with taps in the padding counting zero. Feature maps up to 8 × 8 are simulated pixel by pixel, so borders and padding are exact; larger ones as an 8 × 8 patch that wraps around, where each value takes the window of a random actual output pixel so the effect of the edges is kept. BatchNorm2d normalizes each channel (removing the offset a channel's weights give it, keeping each example's own share), and pooling, upsampling, joins and flatten are applied to the simulated values. A map flattened into a sequence (ViT's patches) is simulated pixel by pixel, every patch its own position (up to 256), so attention sees the real sequence length; a class token starts at zeros, and selecting a position passes on only its gradient.

Checked against real PyTorch (several fresh inits, hooks on every `nn.Linear`): within a few percent for the MLP examples and across every init, norm and activation; for the transformers (GPT-2, and BERT and Llama at reduced width) forward values are within about 10% and gradients mostly within a few percent, with the largest gaps, up to about 35%, on the query and key gradients of deeper layers. For the image models (hooks on every `nn.Linear` and `nn.Conv2d`, BatchNorm in training mode): the basic CNN and ResNet-50 forward values have a median error of about 1% (largest about 10%), YOLOv3 about 1% (largest 4%); gradients have a median of about 5–8% (largest about 15%, through the deepest residual stacks). ViT-B/16: forward values within 12% (median 0.2%), gradients a median of 8%, largest 30% (query gradients, as for the transformers). `python tools/measure_variance.py model.py ClassName "8, 3, 224, 224"` measures a model for comparison. Big models are worked out in a background worker so the page stays responsive (Llama 2 7B and Mamba-130M take a few seconds, ViT-B/16 about ten).

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

## License

MIT, see `LICENSE`.
