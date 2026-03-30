"""Run the MNIST dead ReLU recovery experiment across multiple seeds.

Usage:
    python run.py                           # 5 seeds, 40 epochs
    python run.py --seeds 42 43 --epochs 5  # quick smoke test
    python run.py --save-models             # also save final weights as safetensors
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from WeightDecomp import DecomposedMLP
from WeightDecomp.train_mnist import train_epoch, evaluate, reset_factor_optimizer_state

from utils import count_dead_neurons, kill_neurons, dead_counts_to_json


def run_branch(model, train_loader, test_loader, device, epochs,
               ranks=None, merge_resplit_every=None, lr=1e-3):
    """Train a (deepcopied) model and record per-epoch statistics.

    Returns dict with keys: dead_counts, train_losses, test_accs
    """
    model = copy.deepcopy(model)
    if ranks:
        model.split_all(ranks)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dead_counts = [dead_counts_to_json(count_dead_neurons(model, train_loader, device))]
    train_losses = []
    test_accs = []

    for epoch in range(1, epochs + 1):
        if (merge_resplit_every and ranks
                and epoch > 1 and (epoch - 1) % merge_resplit_every == 0):
            model.merge_all(rerandomize_B=True)
            reset_factor_optimizer_state(optimizer, model)

        loss, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)

        train_losses.append(loss)
        test_accs.append(acc)
        dead_counts.append(
            dead_counts_to_json(count_dead_neurons(model, train_loader, device))
        )

        if epoch % 10 == 0 or epoch == 1:
            dc = dead_counts[-1]
            dc_str = ", ".join(f"L{k}: {v[0]}/{v[1]}" for k, v in dc.items())
            print(f"    Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Test: {100*acc:.2f}% | Dead: [{dc_str}]")

    return model, {
        "dead_counts": dead_counts,
        "train_losses": train_losses,
        "test_accs": test_accs,
    }


def run_seed(seed, args, train_loader, test_loader, device):
    """Run baseline + split for a single seed. Saves results to disk."""
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "seed": seed,
        "layer_sizes": [784] + args.hidden_sizes + [10],
        "ranks": args.ranks,
        "warmup_epochs": args.warmup_epochs,
        "post_epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "merge_resplit_every": args.merge_resplit_every,
        "kill_frac": args.kill_frac,
        "kill_bias": args.kill_bias,
    }
    (seed_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Build model, kill neurons
    torch.manual_seed(seed)
    layer_sizes = config["layer_sizes"]
    base_model = DecomposedMLP(layer_sizes).to(device)
    killed = kill_neurons(base_model, frac=args.kill_frac,
                          bias_val=args.kill_bias, seed=seed + 1000)

    # Warmup
    optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    warmup_dead = [dead_counts_to_json(
        count_dead_neurons(base_model, train_loader, device)
    )]
    warmup_losses = []
    warmup_accs = []

    print(f"  Warmup ({args.warmup_epochs} epochs)")
    for epoch in range(1, args.warmup_epochs + 1):
        loss, _ = train_epoch(base_model, train_loader, optimizer, criterion, device)
        acc = evaluate(base_model, test_loader, device)
        warmup_losses.append(loss)
        warmup_accs.append(acc)
        warmup_dead.append(dead_counts_to_json(
            count_dead_neurons(base_model, train_loader, device)
        ))
        print(f"    Epoch {epoch} | Loss: {loss:.4f} | Test: {100*acc:.2f}%")

    warmup_data = {
        "epochs": args.warmup_epochs,
        "dead_counts": warmup_dead,
        "train_losses": warmup_losses,
        "test_accs": warmup_accs,
        "killed_indices": killed,
    }

    # Baseline
    print(f"  Baseline ({args.epochs} epochs)")
    baseline_model, baseline_post = run_branch(
        base_model, train_loader, test_loader, device,
        epochs=args.epochs, ranks=None, lr=args.lr,
    )
    baseline_data = {"warmup": warmup_data, "post": baseline_post}
    (seed_dir / "baseline.json").write_text(json.dumps(baseline_data, indent=2))

    # Split
    print(f"  Split ranks={args.ranks} ({args.epochs} epochs, "
          f"merge every {args.merge_resplit_every})")
    split_model, split_post = run_branch(
        base_model, train_loader, test_loader, device,
        epochs=args.epochs, ranks=args.ranks,
        merge_resplit_every=args.merge_resplit_every, lr=args.lr,
    )
    split_data = {"warmup": warmup_data, "post": split_post}
    (seed_dir / "split.json").write_text(json.dumps(split_data, indent=2))

    # Save models
    if args.save_models:
        from safetensors.torch import save_model
        save_model(baseline_model, str(seed_dir / "baseline_model.safetensors"))
        save_model(split_model, str(seed_dir / "split_model.safetensors"))
        print("  Models saved.")

    print(f"  Results saved to {seed_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="MNIST dead ReLU recovery experiment"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--merge-resplit-every", type=int, default=10)
    parser.add_argument("--kill-frac", type=float, default=0.5)
    parser.add_argument("--kill-bias", type=float, default=-5.0)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
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

    for seed in args.seeds:
        run_seed(seed, args, train_loader, test_loader, device)

    print(f"\nAll seeds complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
