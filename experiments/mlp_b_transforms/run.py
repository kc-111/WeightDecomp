"""Test all B-transform nonlinearities on a simple MLP (CPU-friendly).

Compares dead neuron revival and optimization speed across:
  - identity:  W + B @ C  (standard bilinear)
  - centered:  W + (B - mean(B)) @ C  (opens B-channel)
  - cayley:    W + Cayley(B) @ C  (orthogonal, dense coupling)
  - softplus:  W + softplus(B) @ C  (positive, no absorbing states)
  - sigmoid:   W + sigmoid(B) @ C  (bounded, position-dependent)
  - tanh:      W + tanh(B) @ C  (bounded, zero-centered)

Usage:
    python experiments/mlp_b_transforms/run.py                    # default
    python experiments/mlp_b_transforms/run.py --epochs 10        # quick
    python experiments/mlp_b_transforms/run.py --device cuda      # GPU
    python experiments/mlp_b_transforms/run.py --transforms identity centered cayley
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from WeightDecomp.decomposed_linear import DecomposedLinear


# ========== B-Transform Implementations ==========

def _cayley_square(M):
    """Cayley transform on square matrix: (I - S)^{-1}(I + S), S = skew(M)."""
    S = (M - M.T) / 2
    r = S.shape[0]
    I = torch.eye(r, device=S.device, dtype=S.dtype)
    return torch.linalg.solve(I - S, I + S)


def _cayley_rect(B):
    """Cayley on rectangular B (m×r): apply Cayley to the r×r inner part.
    B_out = B @ Cayley(B^T B), orthogonalizes the column space."""
    gram = B.T @ B  # (r, r)
    return B @ _cayley_square(gram)


B_TRANSFORMS = {
    "identity":  lambda B: B,
    "centered":  lambda B: B - B.mean(dim=0, keepdim=True),
    "cayley":    lambda B: _cayley_rect(B),
    "softplus":  lambda B: F.softplus(B),
    "sigmoid":   lambda B: torch.sigmoid(B),
    "tanh":      lambda B: torch.tanh(B),
}


class TransformedDecomposedLinear(DecomposedLinear):
    """DecomposedLinear with configurable B transform.

    W_eff = W + Σ φ(B_i) @ C_i  where φ is the chosen transform.
    """

    def __init__(self, in_features, out_features, bias=True, ranks=None,
                 b_transform="identity"):
        self._b_transform_name = b_transform
        self._b_transform_fn = B_TRANSFORMS[b_transform]
        super().__init__(in_features, out_features, bias=bias, ranks=ranks)

    def effective_weight(self):
        W_eff = self.W
        for B, C in zip(self.Bs, self.Cs):
            W_eff = W_eff + self._b_transform_fn(B) @ C
        return W_eff

    def merge(self, rerandomize_B=True):
        with torch.no_grad():
            for B, C in zip(self.Bs, self.Cs):
                self.W.add_(self._b_transform_fn(B) @ C)
                C.zero_()
            if rerandomize_B:
                for B in self.Bs:
                    nn.init.kaiming_uniform_(B, a=math.sqrt(5))


class TransformedMLP(nn.Module):
    """MLP using TransformedDecomposedLinear layers."""

    def __init__(self, layer_sizes, b_transform="identity"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                TransformedDecomposedLinear(
                    layer_sizes[i], layer_sizes[i + 1],
                    b_transform=b_transform
                )
            )

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


# ========== Utilities ==========

@torch.no_grad()
def count_dead(model, loader, device, max_batches=50):
    """Count dead neurons per hidden layer."""
    model.eval()
    layers = model.decomposed_layers()[:-1]
    max_preact = [torch.full((l.out_features,), -float("inf"), device=device)
                  for l in layers]
    for bi, (img, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = img.to(device).view(img.size(0), -1)
        for i, layer in enumerate(layers):
            preact = layer(x)
            max_preact[i] = torch.maximum(max_preact[i], preact.max(dim=0).values)
            x = F.relu(preact)
    return {i: (int((mp < 0).sum()), mp.numel()) for i, mp in enumerate(max_preact)}


def kill_neurons(model, frac=0.5, bias_val=-5.0, seed=123):
    rng = torch.Generator().manual_seed(seed)
    for layer in model.decomposed_layers()[:-1]:
        n = layer.out_features
        k = int(n * frac)
        idx = torch.randperm(n, generator=rng)[:k]
        with torch.no_grad():
            layer.bias.data[idx] = bias_val


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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for img, lab in loader:
        img, lab = img.to(device), lab.to(device)
        correct += model(img).argmax(1).eq(lab).sum().item()
        total += lab.size(0)
    return correct / total


def reset_opt(optimizer, model):
    from WeightDecomp.decomposed_linear import DecomposedLinear
    for module in model.modules():
        if isinstance(module, DecomposedLinear):
            for param in list(module.Bs.parameters()) + list(module.Cs.parameters()):
                if param in optimizer.state:
                    del optimizer.state[param]


# ========== Main ==========

def run_experiment(b_transform, args, train_loader, test_loader, device):
    """Run one experiment with a given B transform."""
    torch.manual_seed(args.seed)
    model = TransformedMLP(args.layer_sizes, b_transform=b_transform).to(device)
    kill_neurons(model, frac=args.kill_frac, bias_val=args.kill_bias, seed=123)

    m = copy.deepcopy(model)
    m.split_all(args.ranks)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, args.epochs + 1):
        if args.merge_every > 0 and ep > 1 and (ep - 1) % args.merge_every == 0:
            m.merge_all(rerandomize_B=True)
            reset_opt(opt, m)

        loss, train_acc = train_epoch(m, train_loader, opt, crit, device,
                                      max_batches=args.max_batches)
        if ep in args.checkpoints or ep == args.epochs:
            test_acc = evaluate(m, test_loader, device)
            dead = count_dead(m, train_loader, device, max_batches=30)
            total_dead = sum(d for d, _ in dead.values())
            total_neurons = sum(t for _, t in dead.values())
            history.append({
                "epoch": ep, "loss": loss, "train_acc": train_acc,
                "test_acc": test_acc, "dead": total_dead, "total": total_neurons,
                "per_layer": [(d, t) for d, t in dead.values()],
            })
    return history


def main():
    parser = argparse.ArgumentParser(description="Test B-transform nonlinearities on MLP")
    parser.add_argument("--layer-sizes", nargs="+", type=int, default=[784, 64, 64, 64, 64, 10])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--merge-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=100,
                        help="Batches per epoch (100 = ~12K samples, fast on CPU)")
    parser.add_argument("--kill-frac", type=float, default=0.5)
    parser.add_argument("--kill-bias", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--transforms", nargs="+", default=list(B_TRANSFORMS.keys()),
                        help="Which transforms to test")
    args = parser.parse_args()

    args.checkpoints = set([1, 5, 10, 15, 20, 25, 30, 40, 50] +
                           list(range(5, args.epochs + 1, 5)))

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Architecture: {args.layer_sizes}")
    print(f"Ranks: {args.ranks}, merge every {args.merge_every}")
    print(f"Epochs: {args.epochs}, {args.max_batches} batches/epoch")
    print(f"Transforms: {args.transforms}")
    print()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(root="data", train=True,
                                           transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root="data", train=False,
                                          transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)

    # Also run baseline (no factors)
    print("Running: no_factors", flush=True)
    torch.manual_seed(args.seed)
    base_model = TransformedMLP(args.layer_sizes, b_transform="identity").to(device)
    kill_neurons(base_model, frac=args.kill_frac, bias_val=args.kill_bias, seed=123)
    m0 = copy.deepcopy(base_model)
    opt0 = torch.optim.Adam(m0.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    baseline_history = []
    for ep in range(1, args.epochs + 1):
        loss, _ = train_epoch(m0, train_loader, opt0, crit, device,
                              max_batches=args.max_batches)
        if ep in args.checkpoints or ep == args.epochs:
            acc = evaluate(m0, test_loader, device)
            dead = count_dead(m0, train_loader, device, max_batches=30)
            td = sum(d for d, _ in dead.values())
            tt = sum(t for _, t in dead.values())
            baseline_history.append({"epoch": ep, "loss": loss, "test_acc": acc,
                                     "dead": td, "total": tt})

    results = {"no_factors": baseline_history}
    for bt in args.transforms:
        print(f"Running: {bt}", flush=True)
        results[bt] = run_experiment(bt, args, train_loader, test_loader, device)

    # Print results
    all_names = ["no_factors"] + args.transforms
    checkpoints = sorted(set(h["epoch"] for h in results[all_names[0]]))

    print()
    print("=" * 100)
    print(f"{'Transform':<12}", end="")
    for ep in checkpoints:
        print(f" |  ep {ep:>2}            ", end="")
    print()
    print(f"{'':12}", end="")
    for _ in checkpoints:
        print(f" | {'loss':>6} {'acc':>5} {'dead':>5}", end="")
    print()
    print("-" * 100)

    for name in all_names:
        hist = results[name]
        print(f"{name:<12}", end="")
        for h in hist:
            print(f" | {h['loss']:6.3f} {100*h['test_acc']:4.1f}% {h['dead']:>3}  ", end="")
        print()

    # Summary at final epoch
    print()
    print("FINAL RANKING (sorted by dead count, then loss):")
    print(f"{'Transform':<12} {'Loss':>8} {'Acc':>7} {'Dead':>8} {'Revived':>8}")
    print("-" * 48)
    final = [(name, results[name][-1]) for name in all_names]
    final.sort(key=lambda x: (x[1]["dead"], x[1]["loss"]))
    for name, h in final:
        total = h.get("total", 0)
        revived = total - h["dead"] if total else 0
        print(f"{name:<12} {h['loss']:8.4f} {100*h['test_acc']:6.1f}% "
              f"{h['dead']:>4}/{total:<3} {revived:>4}")


    # Save results
    import json
    out_path = Path(__file__).parent / "results.json"
    # Convert to JSON-safe
    json_results = {}
    for name, hist in results.items():
        json_results[name] = hist
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": json_results}, f, indent=2,
                  default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
