# Torch Canvas for agents

How a model like Claude (or any program) can open a network in Torch Canvas, read it back, and hand it to a person. Site: <https://jessetnroberts.github.io/torch-canvas/>

## Links that carry a model

A link opens the model in a new tab, laid out and ready to edit. The model is inside the link, after the `#`, so nothing is uploaded: the part after `#` never leaves the browser. The text is read the way a paste is (parsed, never run), and opening a link never replaces a saved tab.

| Link | What follows the `=` | Use it for |
|---|---|---|
| `…/torch-canvas/#code=…` | the Python, URL-encoded (`encodeURIComponent`, `urllib.parse.quote(src, safe='')`) | writing a link by hand; short models |
| `…/torch-canvas/#zip=…` | the text compressed with raw DEFLATE, then base64url without padding | what **Copy link** makes; anything longer |
| `…/torch-canvas/#url=…` | an http(s) link to a `.py` or `.json` file | a model too big for a link: a gist's **Raw** link, or a GitHub file (`github.com/…/blob/…` is turned into its raw address) |

The text can be PyTorch code, a torch.fx export from `export_graph.py`, or a Torch Canvas graph (JSON, which keeps where everything sits, nicknames, notes and saved views). Any of them works in `#code=`, `#zip=` and `#url=`.

Making a `#zip=` link in Python:

```python
import base64, zlib
def torch_canvas_link(text):
    c = zlib.compressobj(9, zlib.DEFLATED, -15)  # -15: raw DEFLATE, no header
    z = c.compress(text.encode()) + c.flush()
    return "https://jessetnroberts.github.io/torch-canvas/#zip=" + base64.urlsafe_b64encode(z).decode().rstrip("=")
```

and reading one back: `zlib.decompress(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)), -15).decode()`.

GPT-2's code is a link of about 1.7 KB as `#zip=`. Some chat apps and email clients cut long links short; above a few KB, prefer `#url=`.

In the app, **Copy link ▾** (above the code) makes either kind: **Link with the code** (shortest; opens laid out automatically) or **Link with this drawing** (the graph, so it opens exactly as drawn).

## Writing code Torch Canvas reads well

The surest template is the code Torch Canvas writes itself: open a similar example and copy its code. In short:

- One class per module, subclassing `nn.Module`, with sizes as constructor arguments with defaults (`def __init__(self, d=768, n_head=12)`). The top-level model is the class with a `forward()` that no other class constructs (the last one, if there are several).
- Give the example input a line of its own and assign the call, so the input's shape is read: `x = torch.randn(8, 64, 768)` then `y = model(x)` inside `if __name__ == "__main__":`. Without it, an input is assumed to be (batch, width of the first layer).
- Layers that come in as real layers: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.RMSNorm`, `nn.GroupNorm`, `nn.BatchNorm1d/2d`, `nn.InstanceNorm1d`, `nn.Conv1d`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`, `nn.Upsample`/`F.interpolate`, `nn.Flatten`, `nn.Dropout`, `nn.Parameter`, and the activations (`ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, softplus) as modules, `F.` functions or `torch.` functions.
- Structure: `nn.Sequential`, `nn.ModuleList` (also built with a list comprehension) and `for` loops over it, custom submodules (they become blocks), `x + y`, `x * y`, `x @ y`, constants (`q * (self.d / self.n_head) ** -0.5`), `torch.cat`, `torch.stack`.
- Rearranging a tensor's axes: `.unflatten`, `.flatten`, `.transpose`, `.permute`, `.squeeze`, `.unsqueeze` with constant arguments become Reshape nodes. Split heads with `self.q(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)` and merge them with `y.transpose(1, 2).flatten(2)`.
- Attention: separate `q`, `k`, `v` projections, `q @ k.transpose(-2, -1)`, `torch.softmax(…, dim=-1)`, `att @ v`. A causal mask is the helper Torch Canvas writes (`causal_mask(scores)`, in the GPT-2 example).
- Initializers: `nn.init.normal_`, `trunc_normal_`, `zeros_`, `orthogonal_`, `xavier_uniform_`/`xavier_normal_` and `kaiming_uniform_`/`kaiming_normal_`, applied to a layer's `.weight` or `.bias` in `__init__`.
- A `loss()` and `predict()` method in the form Torch Canvas writes them are read back as losses.

What doesn't come through as layers (it is kept as a custom op, so the code still regenerates, and a note says so):

- `.view(…)` or `.reshape(…)` with sizes read from `x.shape`, `einops.rearrange`, `.mean()`, and other tensor methods. Use the rearrangement methods above.
- Unpacking a split: `q, k, v = self.qkv(x).chunk(3, dim=-1)` gives custom ops that aren't tensors further on. Use three projections.
- `F.scaled_dot_product_attention`: write the attention out as above.

For a model the reader can't follow, `export_graph.py` (torch.fx) writes JSON that opens the same way: see the guide's Import section.

## The page's interface: `window.torchCanvas`

An agent driving the page (a browser tool, a Playwright script) can call these instead of clicking. Everything returns copies, so nothing held can change the model behind the page's back.

| Call | Does |
|---|---|
| `await torchCanvas.load(text)` | reads code, an fx export or graph JSON into a new tab. Returns `{ok: true, …summary}` or `{ok: false, error}` |
| `await torchCanvas.loadUrl(url)` | the same, from a file at a link |
| `torchCanvas.code()` | the PyTorch code of the open tab |
| `torchCanvas.summary()` | `{name, params, paramsExact, inputs, outputs, layers, blocks, sizes, views, errors, notes}`; shapes list sizes, numbers where they're known (`e` is the batch) |
| `torchCanvas.graph()` | the graph JSON (what a drawing link carries) |
| `torchCanvas.svg()` | the diagram as SVG text, as **Export SVG** saves it |
| `await torchCanvas.link({layout})` | a `#zip=` link to the open tab: the code, or with `layout: true` the drawing |
| `torchCanvas.tabs()`, `select(idOrName)`, `await close(idOrName)` | the open tabs |
| `torchCanvas.examples()`, `openExample(name)` | the example networks and library blocks (`openExample('gpt2')`) |
| `torchCanvas.explode(view?)`, `views()`, `applyView(name)` | Explode blocks and saved views |
| `torchCanvas.setOptions({namedSizes, shapeComments})` | how the code is written |

`torchCanvas.help` lists the calls; `torchCanvas.version` is 1.

A round trip with a person: they press **Copy link** and paste it to the agent; the agent reads the model out of the link (decode the `#zip=` as above), changes the code, and sends back a new link.
