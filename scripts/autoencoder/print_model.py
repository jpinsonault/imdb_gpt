
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from typing import List

def print_model_layers_with_shapes(model: nn.Module, sample_movie_inputs: List[torch.Tensor]):
    model.eval()
    rows = []
    hooks = []

    def shape_of(x):
        if isinstance(x, torch.Tensor):
            return tuple(x.size())
        if isinstance(x, (list, tuple)):
            return [shape_of(t) for t in x]
        return type(x).__name__

    want = (
        nn.Conv1d,
        nn.ConvTranspose1d,
        nn.Linear,
        nn.LayerNorm,
        nn.Embedding,
        nn.GELU,
        nn.ReLU,
        nn.MultiheadAttention,
        nn.TransformerDecoder,
        nn.TransformerDecoderLayer,
        nn.Flatten,
        nn.Identity,
    )

    for name, mod in model.named_modules():
        if name == "":
            continue
        if not isinstance(mod, want):
            continue
        def make_hook(nm, m):
            def _h(module, inputs, output):
                in_s = [shape_of(i) for i in inputs]
                out_s = shape_of(output)
                rows.append((nm, module.__class__.__name__, in_s, out_s))
            return _h
        hooks.append(mod.register_forward_hook(make_hook(name, mod)))

    with torch.no_grad():
        _ = model(sample_movie_inputs)

    for h in hooks:
        h.remove()

    width_name = max(12, min(60, max((len(n) for n, _, _, _ in rows), default=12)))
    print("-" * (width_name + 68))
    print(f"{'layer':<{width_name}}  {'type':<28}  {'input':<18}  {'output':<18}")
    print("-" * (width_name + 68))
    for nm, typ, ins, out in rows:
        ins_str = str(ins[:1])[0:18]
        out_str = str(out)[0:18]
        print(f"{nm:<{width_name}}  {typ:<28}  {ins_str:<18}  {out_str:<18}")
    print("-" * (width_name + 68))
    model.train()
