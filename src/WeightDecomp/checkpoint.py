"""Pickle-free checkpointing using safetensors + JSON.

Model weights → safetensors (via save_model/load_model for shared tensors)
Optimizer state tensors → safetensors
Optimizer config + training metadata → JSON
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file, save_model, load_model


def save_checkpoint(path, model, optimizer, metadata=None):
    """Save model + optimizer + metadata without pickle.

    Args:
        path: checkpoint directory
        model: nn.Module
        optimizer: torch.optim.Optimizer
        metadata: dict of JSON-serializable training state (epoch, iter, etc.)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Model weights (handles shared tensors)
    save_model(model, str(path / "model.safetensors"))

    # Optimizer state: separate tensors from config
    opt_sd = optimizer.state_dict()

    # Flatten optimizer state tensors into {key: tensor}
    opt_tensors = {}
    opt_meta = {"param_groups": opt_sd["param_groups"], "state_keys": {}}
    for param_id, state in opt_sd["state"].items():
        opt_meta["state_keys"][str(param_id)] = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                flat_key = f"state.{param_id}.{k}"
                opt_tensors[flat_key] = v
                opt_meta["state_keys"][str(param_id)][k] = "tensor"
            else:
                opt_meta["state_keys"][str(param_id)][k] = v

    if opt_tensors:
        save_file(opt_tensors, str(path / "optimizer.safetensors"))
    (path / "optimizer_meta.json").write_text(
        json.dumps(opt_meta, indent=2, default=str))

    # Training metadata
    if metadata is not None:
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str))

    print(f"  Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer, device="cpu"):
    """Load model + optimizer without pickle. Returns metadata dict."""
    path = Path(path)

    # Model
    load_model(model, str(path / "model.safetensors"), device=str(device))

    # Optimizer state
    opt_meta_path = path / "optimizer_meta.json"
    opt_tensor_path = path / "optimizer.safetensors"

    if opt_meta_path.exists():
        opt_meta = json.loads(opt_meta_path.read_text())

        opt_tensors = {}
        if opt_tensor_path.exists():
            opt_tensors = load_file(str(opt_tensor_path), device=str(device))

        # Reconstruct optimizer state dict
        state = {}
        for param_id_str, keys_info in opt_meta["state_keys"].items():
            param_id = int(param_id_str)
            state[param_id] = {}
            for k, v in keys_info.items():
                if v == "tensor":
                    flat_key = f"state.{param_id}.{k}"
                    state[param_id][k] = opt_tensors[flat_key]
                else:
                    state[param_id][k] = v

        optimizer.load_state_dict({
            "state": state,
            "param_groups": opt_meta["param_groups"],
        })

    # Metadata
    metadata = {}
    meta_path = path / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    print(f"  Resumed from {path}")
    return metadata
