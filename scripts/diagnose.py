"""Quick diagnostic: create model, kill neurons, train a few epochs, print diagnostics.

Runs in ~2 minutes on GPU. Uses 20 batches per epoch for speed.
Usage: python scripts/diagnose.py [--n-layers 60] [--d-model 64] [--epochs 5]
"""

import argparse
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from WeightDecomp import DecomposedViT
from WeightDecomp.diagnostics import (
    count_dead_neurons, gradient_flow_summary, DiagnosticTracker
)
from WeightDecomp.train_mnist import train_epoch, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--shared-ranks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--local-ranks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--merge-every", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--kill-frac", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data (small subset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(root="data", train=True,
                                           transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root="data", train=False,
                                          transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)

    # Model
    torch.manual_seed(args.seed)
    model = DecomposedViT(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, skip=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.n_layers} layers, d={args.d_model}, ff={args.d_ff}")
    print(f"Parameters: {n_params:,}")

    # Kill neurons
    rng = torch.Generator().manual_seed(123)
    for layer in model.ffn_layers():
        n = layer.out_features
        k = int(n * args.kill_frac)
        idx = torch.randperm(n, generator=rng)[:k]
        with torch.no_grad():
            layer.bias.data[idx] = -5.0

    dead0 = count_dead_neurons(model, train_loader, device, args.max_batches)
    print(f"After kill: {model.dead_neuron_summary(dead0)}")

    # Split
    model.split_all(shared_ranks=args.shared_ranks, local_ranks=args.local_ranks)
    n_params_split = sum(p.numel() for p in model.parameters())
    print(f"After split: {n_params_split:,} params (+{n_params_split - n_params:,})")
    print(f"Shared ranks: {args.shared_ranks}, Local ranks: {args.local_ranks}")
    print(f"Scopes: {len(model.scopes)} (one per block)\n")

    # Train with diagnostics
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    tracker = DiagnosticTracker(model, train_loader, device, args.max_batches)

    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and (epoch - 1) % args.merge_every == 0:
            model.merge_all(rerandomize_B=True)
            model.reset_factor_optimizer_state(optimizer)
            print(f"  [Merge + resplit at epoch {epoch}]")

        # Train (limited batches)
        model.train()
        total_loss = 0
        n = 0
        for bi, (img, lab) in enumerate(train_loader):
            if bi >= args.max_batches:
                break
            img, lab = img.to(device), lab.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), lab)
            loss.backward()

            # Print gradient flow on first and last epoch
            if bi == 0 and epoch in (1, args.epochs):
                print(f"\n--- Gradient flow (epoch {epoch}, batch 0) ---")
                print(gradient_flow_summary(model))
                print()

            optimizer.step()
            total_loss += loss.item() * img.size(0)
            n += img.size(0)

        train_loss = total_loss / n
        test_acc = evaluate(model, test_loader, device)
        entry = tracker.checkpoint(epoch, train_loss, test_acc)
        summary = model.dead_neuron_summary(entry["dead_counts"])
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Acc: {100*test_acc:.1f}% | {summary}")

    print("\n=== Summary ===")
    tracker.print_summary()


if __name__ == "__main__":
    main()
