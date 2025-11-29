# scripts/autoencoder/print_model.py

import torch
import torch.nn as nn


def _shape_tree(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_tree(t) for t in x]
    if isinstance(x, dict):
        return {k: _shape_tree(v) for k, v in x.items()}
    return type(x).__name__


def _compact_shape(x, max_items=3):
    if isinstance(x, tuple):
        return "(" + ",".join(str(d) for d in x) + ")"
    if isinstance(x, list):
        parts = [_compact_shape(t, max_items=max_items) for t in x[:max_items]]
        if len(x) > max_items:
            parts.append("...")
        return "[" + ", ".join(parts) + "]"
    if isinstance(x, dict):
        items = list(x.items())
        parts = []
        for k, v in items[:max_items]:
            parts.append(f"{k}={_compact_shape(v, max_items=max_items)}")
        if len(items) > max_items:
            parts.append("...")
        return "{" + ", ".join(parts) + "}"
    return str(x)


def _count_params(module: nn.Module):
    total = 0
    trainable = 0
    for p in module.parameters(recurse=False):
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _get_detailed_type(module: nn.Module) -> str:
    """Helper to add dimensions to the type string."""
    base = module.__class__.__name__
    if isinstance(module, nn.Embedding):
        return f"{base}({module.num_embeddings}, {module.embedding_dim})"
    if isinstance(module, nn.Linear):
        return f"{base}({module.in_features}->{module.out_features})"
    return base


def summarize_model(
    model: nn.Module,
    sample_inputs,
    max_depth: int | None = None,
    print_fn=print,
):
    named = {m: name for name, m in model.named_modules()}
    rows = []
    seen = set()
    
    # We use a mutable list to act as a counter in the hook
    exec_order = [0]

    def hook(module, inputs, output):
        mid = id(module)
        if mid in seen:
            return
        seen.add(mid)

        name = named.get(module, "")
        if name == "":
            name = "(root)"
        depth = name.count(".") if name != "(root)" else 0
        if max_depth is not None and depth > max_depth:
            return

        in_shape = _compact_shape(_shape_tree(inputs if len(inputs) != 1 else inputs[0]))
        out_shape = _compact_shape(_shape_tree(output))
        params, trainable = _count_params(module)
        
        type_str = _get_detailed_type(module)

        rows.append(
            {
                "id": exec_order[0],
                "name": name,
                "type": type_str,
                "depth": depth,
                "in": in_shape,
                "out": out_shape,
                "params": params,
                "trainable": trainable,
            }
        )
        exec_order[0] += 1

    # Register hooks
    handles = []
    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    # Run forward pass
    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            _ = model(**sample_inputs)
        elif isinstance(sample_inputs, (list, tuple)):
            _ = model(*sample_inputs)
        else:
            _ = model(sample_inputs)

    # Cleanup
    for h in handles:
        h.remove()

    # Sort by execution order (id) instead of Name
    rows.sort(key=lambda r: r["id"])

    total_params = sum(r["params"] for r in rows)
    total_trainable = sum(r["trainable"] for r in rows)

    header = (
        f"{'Idx':>4}  {'Name':<30}  {'Type':<35}  "
        f"{'Input':<20}  {'Output':<20}  {'Params':>10}  {'Train':>10}"
    )
    sep = "-" * len(header)
    print_fn(sep)
    print_fn(header)
    print_fn(sep)

    for i, r in enumerate(rows):
        indent = "  " * r["depth"]
        name = (indent + r["name"]).strip()
        if len(name) > 30:
            name = name[:27] + "..."
        
        t = r["type"]
        if len(t) > 35:
            t = t[:32] + "..."
            
        inp = r["in"]
        out = r["out"]
        if len(inp) > 20:
            inp = inp[:17] + "..."
        if len(out) > 20:
            out = out[:17] + "..."
            
        print_fn(
            f"{i:4d}  {name:<30}  {t:<35}  "
            f"{inp:<20}  {out:<20}  {r['params']:10d}  {r['trainable']:10d}"
        )

    print_fn(sep)
    print_fn(
        f"{'Total params:':<20} {total_params:d}    "
        f"{'Trainable:':<12} {total_trainable:d}    "
        f"{'Frozen:':<10} {total_params - total_trainable:d}"
    )
    print_fn(sep)

    return {
        "layers": rows,
        "total_params": total_params,
        "total_trainable": total_trainable,
    }


def print_model_summary(model: nn.Module, sample_inputs, max_depth: int | None = None):
    return summarize_model(model, sample_inputs, max_depth=max_depth, print_fn=print)