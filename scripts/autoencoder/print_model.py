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


def summarize_model(
    model: nn.Module,
    sample_inputs,
    max_depth: int | None = None,
    print_fn=print,
):
    named = {m: name for name, m in model.named_modules()}
    rows = []
    seen = set()

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

        rows.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "depth": depth,
                "in": in_shape,
                "out": out_shape,
                "params": params,
                "trainable": trainable,
            }
        )

    handles = []
    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            _ = model(**sample_inputs)
        elif isinstance(sample_inputs, (list, tuple)):
            _ = model(*sample_inputs)
        else:
            _ = model(sample_inputs)

    for h in handles:
        h.remove()

    rows.sort(key=lambda r: (r["depth"], r["name"]))

    total_params = sum(r["params"] for r in rows)
    total_trainable = sum(r["trainable"] for r in rows)

    header = (
        f"{'Idx':>4}  {'Name':<40}  {'Type':<28}  "
        f"{'Input':<28}  {'Output':<28}  {'Params':>10}  {'Train':>10}"
    )
    sep = "-" * len(header)
    print_fn(sep)
    print_fn(header)
    print_fn(sep)

    for i, r in enumerate(rows):
        indent = "  " * r["depth"]
        name = (indent + r["name"]).strip()
        if len(name) > 40:
            name = name[:37] + "..."
        t = r["type"]
        if len(t) > 28:
            t = t[:25] + "..."
        inp = r["in"]
        out = r["out"]
        if len(inp) > 28:
            inp = inp[:25] + "..."
        if len(out) > 28:
            out = out[:25] + "..."
        print_fn(
            f"{i:4d}  {name:<40}  {t:<28}  "
            f"{inp:<28}  {out:<28}  {r['params']:10d}  {r['trainable']:10d}"
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
