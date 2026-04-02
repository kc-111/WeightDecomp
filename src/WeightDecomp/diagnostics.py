"""Diagnostics for weight decomposition training.

Tracks gradient norms, dead neuron counts, and factor evolution.
Designed for fast iteration — uses data subsets by default.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def gradient_norms(model: nn.Module) -> dict[str, dict[str, float]]:
    """Collect per-parameter gradient norms after backward().

    Returns: {param_name: {"norm": float, "type": str}} grouped by layer.
    Call after loss.backward() but before optimizer.step().
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = {
                "norm": param.grad.norm().item(),
                "numel": param.numel(),
            }
    return norms


def gradient_flow_summary(model: nn.Module) -> str:
    """Per-block gradient norm summary for deep models.

    Shows min/max/mean gradient norms across blocks to detect
    vanishing or exploding gradients.
    """
    from .vit import DecomposedViT, SharedFactorLinear
    if not isinstance(model, DecomposedViT):
        return "gradient_flow_summary requires DecomposedViT"

    lines = []
    for i, block in enumerate(model.blocks):
        block_norms = []
        for name, param in block.named_parameters():
            if param.grad is not None:
                block_norms.append(param.grad.norm().item())
        if block_norms:
            mn, mx, avg = min(block_norms), max(block_norms), sum(block_norms)/len(block_norms)
            lines.append(f"  block {i:3d}: min={mn:.2e} max={mx:.2e} avg={avg:.2e}")
        else:
            lines.append(f"  block {i:3d}: no grads")

    # Scope-level stats
    for i, scope in enumerate(model.scopes):
        b_norms = [scope.shared_Bs[j].grad.norm().item()
                   for j in range(len(scope.shared_Bs))
                   if scope.shared_Bs[j].grad is not None]
        if b_norms:
            lines.append(f"  scope {i:3d} B: {' '.join(f'{n:.2e}' for n in b_norms)}")

    return "\n".join(lines)


@torch.no_grad()
def count_dead_neurons(model: nn.Module, loader, device,
                       max_batches: int = 20) -> dict[int, tuple[int, int]]:
    """Count dead neurons in FFN layers using forward hooks.

    Returns: {layer_idx: (num_dead, total)}
    """
    model.eval()
    ffn = model.ffn_layers()
    maxp = [torch.full((l.out_features,), -float("inf"), device=device) for l in ffn]

    def make_hook(idx):
        def hook(module, input, output):
            if output.dim() == 3:
                channel_max = output.amax(dim=(0, 1))
            else:
                channel_max = output.amax(dim=0)
            maxp[idx] = torch.maximum(maxp[idx], channel_max)
        return hook

    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(ffn)]
    for bi, (img, _) in enumerate(loader):
        if bi >= max_batches:
            break
        model(img.to(device))
    for h in handles:
        h.remove()

    return {i: (int((mp < 0).sum()), mp.numel()) for i, mp in enumerate(maxp)}


class DiagnosticTracker:
    """Training-loop-friendly diagnostic tracker.

    Usage:
        tracker = DiagnosticTracker(model, loader, device)
        for epoch in range(epochs):
            loss = train_epoch(...)
            loss_val = loss.backward()  # or however you get loss
            tracker.record_epoch(epoch, train_loss=loss_val)
            if epoch % 5 == 0:
                tracker.checkpoint(epoch)
        tracker.print_summary()
    """

    def __init__(self, model, loader, device, max_batches: int = 20):
        self.model = model
        self.loader = loader
        self.device = device
        self.max_batches = max_batches

        self.history: list[dict] = []

    def checkpoint(self, epoch: int, train_loss: float = 0.0,
                   test_acc: float = 0.0) -> dict:
        """Record dead neurons and factor stats at this epoch."""
        dead = count_dead_neurons(self.model, self.loader, self.device,
                                  self.max_batches)

        n_blocks = len(self.model.blocks) if hasattr(self.model, 'blocks') else 0
        fc1_dead = sum(d for i, (d, _) in dead.items() if i < n_blocks)
        fc1_total = sum(t for i, (_, t) in dead.items() if i < n_blocks)

        # Scope B norms
        b_norms = []
        if hasattr(self.model, 'scopes'):
            for scope in self.model.scopes:
                for j in range(len(scope.shared_Bs)):
                    b_norms.append(scope.shared_Bs[j].norm().item())

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "fc1_dead": fc1_dead,
            "fc1_total": fc1_total,
            "dead_counts": dead,
            "b_norms_avg": sum(b_norms) / len(b_norms) if b_norms else 0,
        }
        self.history.append(entry)
        return entry

    def print_summary(self) -> None:
        """Print compact training summary."""
        print(f"{'Ep':>4} {'Loss':>8} {'Acc':>7} {'fc1 dead':>10} {'B norm':>8}")
        print("-" * 42)
        for h in self.history:
            print(f"{h['epoch']:4d} {h['train_loss']:8.4f} {100*h['test_acc']:6.1f}% "
                  f"{h['fc1_dead']:>4}/{h['fc1_total']:<4} {h['b_norms_avg']:8.3f}")
