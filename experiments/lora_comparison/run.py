"""Compare LoRA vs ReLoRA vs W+BC (ours) for dead neuron revival.

Tests the paper's key claim: LoRA freezes W, so the W-channel gradient is
shut off and dead neurons stay dead permanently. Our approach keeps W trainable,
enabling revival through the coupling matrix P = I + BB^T.

Branches (all from the same killed-neuron checkpoint):
  1. baseline:  no factors, W trainable
  2. lora:      BC factors, W frozen, no merge
  3. relora:    BC factors, W frozen, periodic merge+resplit
  4. ours:      BC factors, W trainable, periodic merge+resplit

Usage:
    python experiments/lora_comparison/run.py                          # default
    python experiments/lora_comparison/run.py --epochs 5               # quick
    python experiments/lora_comparison/run.py --transforms identity sigmoid
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlp_b_transforms"))

from run import (
    TransformedMLP, TransformedDecomposedLinear, B_TRANSFORMS,
    count_dead, kill_neurons, train_epoch, evaluate, reset_opt,
)
from WeightDecomp.decomposed_linear import DecomposedLinear


def freeze_base_weights(model):
    """Freeze W and bias in all DecomposedLinear layers (LoRA mode)."""
    for module in model.modules():
        if isinstance(module, DecomposedLinear):
            module.W.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False


def unfreeze_base_weights(model):
    """Unfreeze W and bias (after merge in ReLoRA)."""
    for module in model.modules():
        if isinstance(module, DecomposedLinear):
            module.W.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True


def run_branch(base_model, train_loader, test_loader, device, args,
               mode="baseline", b_transform="identity"):
    """Run one experimental branch.

    Args:
        mode: "baseline" | "lora" | "relora" | "ours"
    """
    model = copy.deepcopy(base_model)

    has_factors = mode in ("lora", "relora", "ours")
    if has_factors:
        model.split_all(args.ranks)

    if mode in ("lora", "relora"):
        freeze_base_weights(model)

    # Only pass trainable params to optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, args.epochs + 1):
        # Merge+resplit for relora and ours
        if has_factors and args.merge_every > 0 and ep > 1 and (ep - 1) % args.merge_every == 0:
            if mode == "relora":
                # ReLoRA: temporarily unfreeze W for merge, then refreeze
                unfreeze_base_weights(model)
                model.merge_all(rerandomize_B=True)
                freeze_base_weights(model)
                # New optimizer since param requires_grad changed
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(trainable, lr=args.lr)
            elif mode == "ours":
                model.merge_all(rerandomize_B=True)
                reset_opt(optimizer, model)

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
        description="LoRA vs ReLoRA vs W+BC dead neuron revival comparison")
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
    parser.add_argument("--transforms", nargs="+", default=["identity"],
                        help="B-transforms to test (default: identity)")
    args = parser.parse_args()

    args.checkpoints = set(range(1, args.epochs + 1, 5)) | {args.epochs}

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Architecture: {args.layer_sizes}")
    print(f"Ranks: {args.ranks}, merge every {args.merge_every}")
    print(f"Epochs: {args.epochs}, {args.max_batches} batches/epoch")
    print(f"Transforms: {args.transforms}")
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

    for bt in args.transforms:
        print(f"=== B-transform: {bt} ===")

        # Create base model with dead neurons
        torch.manual_seed(args.seed)
        base_model = TransformedMLP(args.layer_sizes, b_transform=bt).to(device)
        kill_neurons(base_model, frac=args.kill_frac, bias_val=args.kill_bias, seed=123)

        dead0 = count_dead(base_model, train_loader, device, max_batches=30)
        total_dead0 = sum(d for d, _ in dead0.values())
        total0 = sum(t for _, t in dead0.values())
        print(f"After kill: {total_dead0}/{total0} dead\n")

        modes = ["baseline", "lora", "relora", "ours"]
        for mode in modes:
            key = f"{bt}_{mode}"
            print(f"  Running: {key}", flush=True)
            results[key] = run_branch(base_model, train_loader, test_loader,
                                      device, args, mode=mode, b_transform=bt)

    # Print results
    print()
    print("=" * 110)
    checkpoints = sorted(args.checkpoints)
    # Show only a few checkpoints for readability
    show_eps = [ep for ep in checkpoints if ep in (1, 5, 10, 20, 30, 40, args.epochs)]

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
                print(f" | {'':>6} {'':>5} {'':>4}", end="")
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
