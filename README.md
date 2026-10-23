<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg">
    <img src="assets/wordmark-light.svg" alt="Torch Canvas" width="440">
  </picture>
</p>

# Torch Canvas

**A visual editor for neural networks that writes PyTorch as you draw, and turns PyTorch back into diagrams.**

Conceived and created by **Jesse Roberts**.

**Try it:** [jessetnroberts.github.io/torch-canvas](https://jessetnroberts.github.io/torch-canvas/), with nothing to install.

## What it does

- **Diagram ⇄ code.** Every edit to the diagram updates clean, runnable PyTorch beside it, and pasting a PyTorch module draws its diagram.
- **Shapes you can see.** Tensor sizes are worked out on every connection, linked sizes change together, and mismatches show in red.
- **Every operation explained.** Selecting a node shows what it computes, as a formula or a plain description. Reshapes are drawn as what they are, rearranging the same values (`e t (h d) → e h t d`), with an optional nickname and note.
- **Reusable blocks.** Group layers into blocks such as a transformer layer or a ResNet bottleneck, repeat them, and keep them in a library.
- **Variance before training.** A plot predicts how the signal and the gradients grow or shrink through the network at initialization.
- **Ready to train.** Export a `train.py` with losses, frozen layers and EMA targets.
- **A library of real models.** It ranges from an MLP to GPT-2, BERT, Llama 2, Qwen2.5-VL, ViT, ResNet-50, YOLOv3, Mamba, U-Net, a VAE, the original Transformer and I-JEPA, each checked against a reference implementation.

## Quick start

Open `index.html` in a browser, or run a local server:

    npm start          # http://localhost:5173

## Documentation

- [Guide](docs/guide.md): the panels, components, editing, sizes, blocks, ties and files.
- [Example models](docs/models.md): what each model is, how it is built and how it was checked.
- [Training and the variance plot](docs/training.md): losses, the `train.py` export, and how the variance plot works.
- [Development and publishing](docs/development.md): building, testing, GitHub Pages and links to examples.

## License

MIT, © 2026 Jesse Roberts. See [LICENSE](LICENSE).
