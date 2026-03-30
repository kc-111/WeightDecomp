"""Shared utilities for the MNIST dead ReLU recovery experiment."""

import torch
import torch.nn.functional as F
from safetensors.torch import save_model, load_model


@torch.no_grad()
def count_dead_neurons(model, loader, device, max_batches=50):
    """Count dead neurons per hidden layer.

    A neuron is dead if its pre-activation is < 0 for every sample checked.

    Returns:
        dict mapping layer index -> (num_dead, total_neurons)
    """
    model.eval()
    layers = model.decomposed_layers()
    hidden_layers = layers[:-1]

    max_preact = [
        torch.full((l.out_features,), -float("inf"), device=device)
        for l in hidden_layers
    ]

    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = images.to(device).view(images.size(0), -1)
        for i, layer in enumerate(hidden_layers):
            preact = layer(x)
            max_preact[i] = torch.maximum(max_preact[i], preact.max(dim=0).values)
            x = F.relu(preact)

    return {
        i: (int((mp < 0).sum()), mp.numel()) for i, mp in enumerate(max_preact)
    }


def kill_neurons(model, frac=0.5, bias_val=-5.0, seed=123):
    """Kill a fraction of neurons in each hidden layer by setting their bias
    to a large negative value.

    Returns:
        list of tensors: killed neuron indices per hidden layer.
    """
    rng = torch.Generator().manual_seed(seed)
    killed = []
    for layer in model.decomposed_layers()[:-1]:
        n = layer.out_features
        n_kill = int(n * frac)
        perm = torch.randperm(n, generator=rng)
        kill_idx = perm[:n_kill]
        with torch.no_grad():
            layer.bias.data[kill_idx] = bias_val
        killed.append(kill_idx.tolist())
    return killed


def dead_counts_to_json(dc):
    """Convert dead counts dict {int: (int, int)} to JSON-safe {str: [int, int]}."""
    return {str(k): list(v) for k, v in dc.items()}
