# export_graph.py: dump an nn.Module as JSON for Torch Canvas
# usage:  from export_graph import export; export(model, torch.randn(8, 784))
#         several inputs: export(model, (src, tgt))
import json, operator
import torch
import torch.fx as fx


class _GroupTracer(fx.Tracer):
    """Records which top-level custom submodule each traced op came from."""
    def __init__(self):
        super().__init__()
        self._stack = []

    def call_module(self, m, forward, args, kwargs):
        self._stack.append((self.path_of_module(m), type(m)))
        try:
            return super().call_module(m, forward, args, kwargs)
        finally:
            self._stack.pop()

    def create_proxy(self, *args, **kwargs):
        proxy = super().create_proxy(*args, **kwargs)
        owner = next(((p, t) for p, t in self._stack
                      if not t.__module__.startswith("torch.nn")), None)
        proxy.node.meta["owner"] = owner
        return proxy


def _target_name(t):
    if isinstance(t, str):
        return t
    if t.__module__ in ("_operator", "operator"):
        return "operator." + t.__name__
    if getattr(torch, getattr(t, "__name__", ""), None) is t:
        return "torch." + t.__name__
    return f"{t.__module__}.{t.__name__}"


def _arg(a):
    if isinstance(a, fx.Node):
        return {"node": a.name}
    if isinstance(a, (int, float, bool, str)) or a is None:
        return a
    if isinstance(a, (tuple, list)):  # torch.cat([a, b]), sizes (2, 3)
        return [_arg(x) for x in a]
    return repr(a)


def _module_info(m):
    info = {"type": type(m).__name__, "repr": "nn." + repr(m)}
    first = lambda v: v[0] if isinstance(v, (tuple, list)) else v
    if isinstance(m, torch.nn.Linear):
        info.update(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
    elif isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        info.update(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=first(m.kernel_size), stride=first(m.stride),
                    padding=first(m.padding) if not isinstance(m.padding, str) else m.padding, bias=m.bias is not None, groups=m.groups)
    elif isinstance(m, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
        info.update(kernel_size=first(m.kernel_size), stride=first(m.stride) if m.stride is not None else None, padding=first(m.padding))
    elif isinstance(m, torch.nn.AdaptiveAvgPool2d):
        info.update(output_size=first(m.output_size))
    elif isinstance(m, torch.nn.Flatten):
        info.update(start_dim=m.start_dim, end_dim=m.end_dim)
    elif isinstance(m, torch.nn.Embedding):
        info.update(num_embeddings=m.num_embeddings, embedding_dim=m.embedding_dim)
    elif hasattr(m, "normalized_shape"):
        info["dim"] = list(m.normalized_shape)[-1]
    elif hasattr(m, "num_features"):
        info["dim"] = m.num_features
    elif hasattr(m, "num_channels"):
        info.update(dim=m.num_channels, num_groups=m.num_groups)
    if hasattr(m, "p") and isinstance(m.p, float):
        info["p"] = m.p
    if hasattr(m, "dim") and isinstance(getattr(m, "dim"), int):
        info["dim"] = m.dim
    return info


def export(model, example=None, path="graph.json"):
    tracer = _GroupTracer()
    graph = tracer.trace(model)
    gm = fx.GraphModule(model, graph)
    if example is not None:
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*(example if isinstance(example, (tuple, list)) else (example,)))
    nodes = []
    for n in gm.graph.nodes:
        rec = {"name": n.name, "op": n.op, "target": _target_name(n.target),
               "kwargs": {k: _arg(v) for k, v in n.kwargs.items()}}
        if n.op == "output":
            outs = n.args[0] if isinstance(n.args[0], (tuple, list)) else [n.args[0]]
            rec["args"] = [_arg(a) for a in outs]
        else:
            rec["args"] = [_arg(a) for a in n.args]
        if n.op == "call_module":
            rec["module"] = _module_info(gm.get_submodule(n.target))
        tm = n.meta.get("tensor_meta")
        if n.op == "placeholder" and tm is not None:
            rec["shape"] = list(tm.shape)
        owner = n.meta.get("owner")
        if owner:
            rec["group"] = {"path": owner[0], "cls": owner[1].__name__}
        nodes.append(rec)
    data = {"name": type(model).__name__, "nodes": nodes}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {path} ({len(nodes)} nodes)")
    return data
