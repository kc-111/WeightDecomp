"""Learning rate sensitivity experiment for MNIST dead ReLU recovery.

Sweeps over multiple learning rates, running baseline vs split for each LR
across multiple seeds. Saves results per (lr, seed) combination.

Usage:
    python run_lr_sweep.py
    python run_lr_sweep.py --lrs 1e-4 5e-4 1e-3 5e-3 --seeds 42 43
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from run import run_branch
from utils import count_dead_neurons, kill_neurons, dead_counts_to_json

from WeightDecomp import DecomposedMLP
from WeightDecomp.train_mnist import train_epoch, evaluate


def run_lr_seed(lr, seed, args, train_loader, test_loader, device):
    """Run baseline + split for a single (lr, seed) pair."""
    lr_label = f"{lr:.0e}"
    run_dir = Path(args.output_dir) / f"lr_{lr_label}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": seed,
        "lr": lr,
        "layer_sizes": [784] + args.hidden_sizes + [10],
        "ranks": args.ranks,
        "warmup_epochs": args.warmup_epochs,
        "post_epochs": args.epochs,
        "batch_size": args.batch_size,
        "merge_resplit_every": args.merge_resplit_every,
        "kill_frac": args.kill_frac,
        "kill_bias": args.kill_bias,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    torch.manual_seed(seed)
    layer_sizes = config["layer_sizes"]
    base_model = DecomposedMLP(layer_sizes).to(device)
    killed = kill_neurons(base_model, frac=args.kill_frac,
                          bias_val=args.kill_bias, seed=seed + 1000)

    # Warmup
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    warmup_dead = [dead_counts_to_json(
        count_dead_neurons(base_model, train_loader, device)
    )]
    warmup_losses = []
    warmup_accs = []

    for epoch in range(1, args.warmup_epochs + 1):
        loss, _ = train_epoch(base_model, train_loader, optimizer, criterion, device)
        acc = evaluate(base_model, test_loader, device)
        warmup_losses.append(loss)
        warmup_accs.append(acc)
        warmup_dead.append(dead_counts_to_json(
            count_dead_neurons(base_model, train_loader, device)
        ))

    warmup_data = {
        "epochs": args.warmup_epochs,
        "dead_counts": warmup_dead,
        "train_losses": warmup_losses,
        "test_accs": warmup_accs,
        "killed_indices": killed,
    }

    # Baseline
    _, baseline_post = run_branch(
        base_model, train_loader, test_loader, device,
        epochs=args.epochs, ranks=None, lr=lr,
    )
    (run_dir / "baseline.json").write_text(json.dumps(
        {"warmup": warmup_data, "post": baseline_post}, indent=2
    ))

    # Split
    _, split_post = run_branch(
        base_model, train_loader, test_loader, device,
        epochs=args.epochs, ranks=args.ranks,
        merge_resplit_every=args.merge_resplit_every, lr=lr,
    )
    (run_dir / "split.json").write_text(json.dumps(
        {"warmup": warmup_data, "post": split_post}, indent=2
    ))

    # Return final stats for summary
    bl_dead_l0 = baseline_post["dead_counts"][-1]["0"][0]
    sp_dead_l0 = split_post["dead_counts"][-1]["0"][0]
    bl_acc = baseline_post["test_accs"][-1]
    sp_acc = split_post["test_accs"][-1]
    return bl_dead_l0, sp_dead_l0, bl_acc, sp_acc


def main():
    parser = argparse.ArgumentParser(
        description="LR sensitivity experiment for dead ReLU recovery"
    )
    parser.add_argument("--lrs", nargs="+", type=float,
                        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--merge-resplit-every", type=int, default=10)
    parser.add_argument("--kill-frac", type=float, default=0.5)
    parser.add_argument("--kill-bias", type=float, default=-5.0)
    parser.add_argument("--output-dir", type=str, default="results_lr_sweep")
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=transform, download=False
    )
    test_set = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=transform, download=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size
    )

    # Run all (lr, seed) combinations
    summary = {}
    for lr in args.lrs:
        lr_label = f"{lr:.0e}"
        print(f"\n{'='*60}")
        print(f"Learning rate: {lr}")
        print(f"{'='*60}")

        lr_results = []
        for seed in args.seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            bl_dead, sp_dead, bl_acc, sp_acc = run_lr_seed(
                lr, seed, args, train_loader, test_loader, device
            )
            lr_results.append({
                "seed": seed,
                "baseline_dead_l0": bl_dead,
                "split_dead_l0": sp_dead,
                "baseline_acc": bl_acc,
                "split_acc": sp_acc,
            })
            print(f"BL dead={bl_dead} acc={100*bl_acc:.1f}% | "
                  f"SP dead={sp_dead} acc={100*sp_acc:.1f}%")

        summary[lr_label] = lr_results

    # Save summary
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_dir / 'summary.json'}")

    # Print summary table
    print(f"\n{'LR':>10s} | {'BL Dead L0':>15s} | {'SP Dead L0':>15s} | "
          f"{'BL Acc':>15s} | {'SP Acc':>15s}")
    print("-" * 80)
    for lr_label, results in summary.items():
        bl_dead = [r["baseline_dead_l0"] for r in results]
        sp_dead = [r["split_dead_l0"] for r in results]
        bl_acc = [r["baseline_acc"] for r in results]
        sp_acc = [r["split_acc"] for r in results]

        import numpy as np
        bl_d_m, bl_d_s = np.mean(bl_dead), np.std(bl_dead)
        sp_d_m, sp_d_s = np.mean(sp_dead), np.std(sp_dead)
        bl_a_m, bl_a_s = np.mean(bl_acc) * 100, np.std(bl_acc) * 100
        sp_a_m, sp_a_s = np.mean(sp_acc) * 100, np.std(sp_acc) * 100

        print(f"{lr_label:>10s} | {bl_d_m:5.1f} +/- {bl_d_s:<4.1f} | "
              f"{sp_d_m:5.1f} +/- {sp_d_s:<4.1f} | "
              f"{bl_a_m:5.1f}% +/- {bl_a_s:<4.1f}% | "
              f"{sp_a_m:5.1f}% +/- {sp_a_s:<4.1f}%")


if __name__ == "__main__":
    main()
