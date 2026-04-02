"""Test shared-B ABC decomposition on MLP for dead neuron revival.

W_eff = W + A @ φ(B_shared) @ C  where:
  - A (out × r): per-layer, handles output dimension
  - B (r × r): SHARED across all layers, configurable transform φ
  - C (r × in): per-layer, handles input dimension, init to 0

The shared B creates cross-layer gradient coupling: gradients from layers
without dead neurons flow through B to help revive dead neurons elsewhere.

Branches:
  1. baseline:       no factors
  2. local_bc:       standard per-layer W + BC (no sharing)
  3. shared_abc:     shared B across all layers, per-layer A and C

Usage:
    python experiments/shared_abc/run.py                    # default
    python experiments/shared_abc/run.py --epochs 5         # quick
    python experiments/shared_abc/run.py --transforms identity centered cayley sigmoid
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlp_b_transforms"))

from run import (
    B_TRANSFORMS, count_dead, kill_neurons, evaluate, reset_opt,
)
from WeightDecomp.decomposed_linear import DecomposedLinear


# ========== B transforms for square (r×r) matrices ==========

def _cayley_square(M):
    S = (M - M.T) / 2
    r = S.shape[0]
    I = torch.eye(r, device=S.device, dtype=S.dtype)
    return torch.linalg.solve(I - S, I + S)


SQUARE_B_TRANSFORMS = {
    "identity":  lambda B: B,
    "centered":  lambda B: B - B.mean(dim=0, keepdim=True),
    "cayley":    lambda B: _cayley_square(B),
    "softplus":  lambda B: F.softplus(B),
    "sigmoid":   lambda B: torch.sigmoid(B),
    "tanh":      lambda B: torch.tanh(B),
}

# A transforms: applied to per-layer A (out × r).
# Cross-row transforms (centered, sigmoid, etc.) open the B-channel for dead neurons.
A_TRANSFORMS = {
    "identity":  lambda A: A,
    "centered":  lambda A: A - A.mean(dim=0, keepdim=True),
    "sigmoid":   lambda A: torch.sigmoid(A),
    "softplus":  lambda A: F.softplus(A),
    "tanh":      lambda A: torch.tanh(A),
}


# ========== ABC Linear with external shared B ==========

class ABCLinear(nn.Module):
    """Linear: W_eff = W + Σ φ_A(A_i) @ φ_B(B_shared_i) @ C_i

    A (out × r) and C (r × in) are per-layer.
    B (r × r) is passed in from external scope.
    φ_A and φ_B are configurable transforms.
    """

    def __init__(self, in_features, out_features, bias=True, a_transform="identity"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._a_transform_fn = A_TRANSFORMS[a_transform]
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.As = nn.ParameterList()
        self.Cs = nn.ParameterList()
        # References to shared transformed Bs (set by SharedABCMLP)
        self._transformed_Bs: list[torch.Tensor] = []
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def add_factor(self, rank):
        dev, dtype = self.W.device, self.W.dtype
        A = nn.Parameter(torch.empty(self.out_features, rank, device=dev, dtype=dtype))
        C = nn.Parameter(torch.zeros(rank, self.in_features, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.As.append(A)
        self.Cs.append(C)

    def remove_factors(self):
        self.As = nn.ParameterList()
        self.Cs = nn.ParameterList()

    def effective_weight(self):
        W_eff = self.W
        for i in range(len(self.As)):
            W_eff = W_eff + self._a_transform_fn(self.As[i]) @ self._transformed_Bs[i] @ self.Cs[i]
        return W_eff

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

    def merge(self, rerandomize_A=True):
        with torch.no_grad():
            for i in range(len(self.As)):
                self.W.add_(self._a_transform_fn(self.As[i]) @ self._transformed_Bs[i] @ self.Cs[i])
                self.Cs[i].zero_()
            if rerandomize_A:
                for i in range(len(self.As)):
                    nn.init.kaiming_uniform_(self.As[i], a=math.sqrt(5))


class SharedABCMLP(nn.Module):
    """MLP with shared B (r×r) across all layers, per-layer A and C.

    W_eff = W + φ_A(A) @ φ_B(B_shared) @ C
    """

    def __init__(self, layer_sizes, b_transform="identity", a_transform="identity"):
        super().__init__()
        self.b_transform_name = b_transform
        self.b_transform_fn = SQUARE_B_TRANSFORMS[b_transform]
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(ABCLinear(layer_sizes[i], layer_sizes[i + 1],
                                        a_transform=a_transform))
        # Shared B parameters (r × r) — created by split_all
        self.shared_Bs = nn.ParameterList()

    def _update_transformed_refs(self):
        """Compute φ(B) and push references to all layers."""
        transformed = [self.b_transform_fn(self.shared_Bs[i])
                       for i in range(len(self.shared_Bs))]
        for layer in self.layers:
            layer._transformed_Bs = transformed

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if len(self.shared_Bs) > 0:
            self._update_transformed_refs()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def decomposed_layers(self):
        return list(self.layers)

    def split_all(self, ranks):
        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.shared_Bs = nn.ParameterList()
        for r in ranks:
            B = nn.Parameter(torch.empty(r, r, device=dev, dtype=dtype))
            nn.init.orthogonal_(B)
            self.shared_Bs.append(B)
        for layer in self.layers:
            layer.remove_factors()
            for r in ranks:
                layer.add_factor(r)

    def merge_all(self, rerandomize_B=True):
        if len(self.shared_Bs) == 0:
            return
        self._update_transformed_refs()
        for layer in self.layers:
            layer.merge(rerandomize_A=True)
        if rerandomize_B:
            with torch.no_grad():
                for i in range(len(self.shared_Bs)):
                    nn.init.orthogonal_(self.shared_Bs[i])


# ========== Also need a local-BC MLP for comparison ==========

class LocalBCLinear(DecomposedLinear):
    """Standard DecomposedLinear with configurable B transform."""

    def __init__(self, in_features, out_features, bias=True, b_transform="identity"):
        self._b_fn = B_TRANSFORMS[b_transform]
        super().__init__(in_features, out_features, bias=bias)

    def effective_weight(self):
        W_eff = self.W
        for B, C in zip(self.Bs, self.Cs):
            W_eff = W_eff + self._b_fn(B) @ C
        return W_eff

    def merge(self, rerandomize_B=True):
        with torch.no_grad():
            for B, C in zip(self.Bs, self.Cs):
                self.W.add_(self._b_fn(B) @ C)
                C.zero_()
            if rerandomize_B:
                for B in self.Bs:
                    nn.init.kaiming_uniform_(B, a=math.sqrt(5))


class LocalBCMLP(nn.Module):
    def __init__(self, layer_sizes, b_transform="identity"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                LocalBCLinear(layer_sizes[i], layer_sizes[i + 1],
                              b_transform=b_transform))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def decomposed_layers(self):
        return list(self.layers)

    def split_all(self, ranks):
        for layer in self.layers:
            layer.split(ranks)

    def merge_all(self, rerandomize_B=True):
        for layer in self.layers:
            layer.merge(rerandomize_B=rerandomize_B)


# ========== Training ==========

def train_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for bi, (img, lab) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img, lab = img.to(device), lab.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
        correct += out.argmax(1).eq(lab).sum().item()
        total += lab.size(0)
    return total_loss / total, correct / total


def reset_optimizer(optimizer, model):
    for module in model.modules():
        for attr in ['Bs', 'Cs', 'As', 'shared_Bs']:
            v = getattr(module, attr, None)
            if isinstance(v, nn.ParameterList):
                for p in v.parameters():
                    if p in optimizer.state:
                        del optimizer.state[p]


# ========== Experiment Runner ==========

def run_branch(base_model, train_loader, test_loader, device, args, mode="baseline"):
    model = copy.deepcopy(base_model)

    if mode != "baseline":
        model.split_all(args.ranks)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []

    for ep in range(1, args.epochs + 1):
        if mode != "baseline" and args.merge_every > 0:
            if ep > 1 and (ep - 1) % args.merge_every == 0:
                model.merge_all(rerandomize_B=True)
                reset_optimizer(optimizer, model)

        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                      device, max_batches=args.max_batches)

        if ep in args.checkpoints or ep == args.epochs:
            test_acc = evaluate(model, test_loader, device)
            dead = count_dead(model, train_loader, device, max_batches=30)
            total_dead = sum(d for d, _ in dead.values())
            total_neurons = sum(t for _, t in dead.values())
            history.append({
                "epoch": ep, "loss": loss, "train_acc": train_acc,
                "test_acc": test_acc, "dead": total_dead, "total": total_neurons,
                "per_layer": [(d, t) for d, t in dead.values()],
            })
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Shared-B ABC vs local BC for dead neuron revival")
    parser.add_argument("--layer-sizes", nargs="+", type=int,
                        default=[784, 64, 64, 64, 64, 10])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--merge-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--kill-frac", type=float, default=0.5)
    parser.add_argument("--kill-bias", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--b-transforms", nargs="+",
                        default=["identity", "centered"],
                        help="B-transforms to test on the shared B (r×r)")
    parser.add_argument("--a-transforms", nargs="+",
                        default=["identity", "centered"],
                        help="A-transforms to test on per-layer A (out×r)")
    args = parser.parse_args()

    args.checkpoints = set(range(1, args.epochs + 1, 5)) | {args.epochs}

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Architecture: {args.layer_sizes}")
    print(f"Ranks: {args.ranks}, merge every {args.merge_every}")
    print(f"Epochs: {args.epochs}, {args.max_batches} batches/epoch")
    print(f"B-transforms: {args.b_transforms}")
    print(f"A-transforms: {args.a_transforms}")
    print()

    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root="data", train=True,
                                           transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root="data", train=False,
                                          transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)

    results = {}

    # Baseline (no factors)
    torch.manual_seed(args.seed)
    base = LocalBCMLP(args.layer_sizes, b_transform="identity").to(device)
    kill_neurons(base, frac=args.kill_frac, bias_val=args.kill_bias, seed=123)
    dead0 = count_dead(base, train_loader, device, max_batches=30)
    print(f"After kill: {sum(d for d,_ in dead0.values())}/{sum(t for _,t in dead0.values())} dead")
    print(f"  Running: baseline", flush=True)
    results["baseline"] = run_branch(base, train_loader, test_loader,
                                     device, args, mode="baseline")

    # Local BC variants (per-layer, no sharing)
    for bt in args.b_transforms:
        key = f"local_B={bt}"
        torch.manual_seed(args.seed)
        model = LocalBCMLP(args.layer_sizes, b_transform=bt).to(device)
        kill_neurons(model, frac=args.kill_frac, bias_val=args.kill_bias, seed=123)
        print(f"  Running: {key}", flush=True)
        results[key] = run_branch(model, train_loader, test_loader,
                                  device, args, mode="split")

    # Shared ABC variants (shared B, per-layer A and C)
    for bt in args.b_transforms:
        for at in args.a_transforms:
            key = f"shared_A={at}_B={bt}"
            torch.manual_seed(args.seed)
            model = SharedABCMLP(args.layer_sizes, b_transform=bt,
                                 a_transform=at).to(device)
            kill_neurons(model, frac=args.kill_frac, bias_val=args.kill_bias, seed=123)
            print(f"  Running: {key}", flush=True)
            results[key] = run_branch(model, train_loader, test_loader,
                                      device, args, mode="split")
    print()

    # Print results
    print("=" * 110)
    show_eps = sorted(ep for ep in args.checkpoints
                      if ep in (1, 5, 10, 20, 30, 40, args.epochs))

    print(f"{'Method':<22}", end="")
    for ep in show_eps:
        print(f" | ep {ep:>2}            ", end="")
    print()
    print(f"{'':22}", end="")
    for _ in show_eps:
        print(f" | {'loss':>6} {'acc':>5} {'dead':>4}", end="")
    print()
    print("-" * 110)

    for key in results:
        hist = results[key]
        print(f"{key:<22}", end="")
        for ep in show_eps:
            match = [h for h in hist if h["epoch"] == ep]
            if match:
                h = match[0]
                print(f" | {h['loss']:6.3f} {100*h['test_acc']:4.1f}% {h['dead']:>4}", end="")
            else:
                print(f" | {'':>17}", end="")
        print()

    # Final ranking
    print()
    print("FINAL RANKING:")
    print(f"{'Method':<22} {'Loss':>8} {'Acc':>7} {'Dead':>8} {'Revived':>8}")
    print("-" * 56)
    final = [(key, results[key][-1]) for key in results]
    final.sort(key=lambda x: (x[1]["dead"], x[1]["loss"]))
    for key, h in final:
        total = h.get("total", 0)
        revived = total - h["dead"]
        print(f"{key:<22} {h['loss']:8.4f} {100*h['test_acc']:6.1f}% "
              f"{h['dead']:>4}/{total:<3} {revived:>4}")

    # Save
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
