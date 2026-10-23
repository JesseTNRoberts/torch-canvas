# Training and the variance plot

Losses, the train.py export, and how the variance plot predicts signal and gradient sizes.

[← Back to the README](../README.md)

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
