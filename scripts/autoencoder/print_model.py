import torch

def _shape_tree(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_tree(t) for t in x]
    return None

def print_model_layers_with_shapes(model, sample_inputs):
    model.eval()
    with torch.no_grad():
        if isinstance(sample_inputs, (list, tuple)) and len(sample_inputs) == 2 and isinstance(sample_inputs[0], list) and isinstance(sample_inputs[1], list):
            out = model(sample_inputs[0], sample_inputs[1])
        else:
            out = model(sample_inputs)
    try:
        ins = _shape_tree(sample_inputs)
        outs = _shape_tree(out)
        print("input_shapes:", ins)
        print("output_shapes:", outs)
    except Exception:
        pass
    return out
