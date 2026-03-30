"""Plot MNIST dead ReLU recovery results across multiple seeds.

Usage:
    python plot.py                          # default results dir
    python plot.py --results-dir results    # explicit path
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_seeds(results_dir):
    """Load baseline.json and split.json from each seed directory.

    Returns:
        list of dicts, each with keys: seed, baseline, split, config
    """
    results_dir = Path(results_dir)
    runs = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        baseline_path = seed_dir / "baseline.json"
        split_path = seed_dir / "split.json"
        config_path = seed_dir / "config.json"
        if not (baseline_path.exists() and split_path.exists()):
            continue
        runs.append({
            "seed": seed_dir.name,
            "baseline": json.loads(baseline_path.read_text()),
            "split": json.loads(split_path.read_text()),
            "config": json.loads(config_path.read_text()) if config_path.exists() else {},
        })
    return runs


def extract_full_timeseries(run_data, key, layer_idx=None):
    """Extract a full timeseries (warmup + post) from a run.

    key: "train_losses", "test_accs", or "dead_counts"
    layer_idx: required if key == "dead_counts", selects which layer
    """
    warmup = run_data["warmup"]
    post = run_data["post"]

    if key == "dead_counts":
        # dead_counts has one extra entry (epoch 0 = initial state)
        warmup_vals = [dc[str(layer_idx)][0] for dc in warmup[key]]
        post_vals = [dc[str(layer_idx)][0] for dc in post[key]]
        return warmup_vals + post_vals
    else:
        return warmup[key] + post[key]


def compute_mean_std(runs, experiment, key, layer_idx=None):
    """Compute mean and std across seeds for a given timeseries."""
    all_series = []
    for run in runs:
        ts = extract_full_timeseries(run[experiment], key, layer_idx)
        all_series.append(ts)

    arr = np.array(all_series)
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for figures (default: results-dir/figures)")
    args = parser.parse_args()

    runs = load_all_seeds(args.results_dir)
    if not runs:
        print(f"No results found in {args.results_dir}/")
        return

    n_seeds = len(runs)
    config = runs[0]["config"]
    warmup_epochs = config.get("warmup_epochs", 2)
    post_epochs = config.get("post_epochs", 40)
    merge_every = config.get("merge_resplit_every", 10)

    print(f"Loaded {n_seeds} seeds from {args.results_dir}/")
    print(f"Config: warmup={warmup_epochs}, post={post_epochs}, "
          f"merge_every={merge_every}, ranks={config.get('ranks')}")

    output_dir = Path(args.output) if args.output else Path(args.results_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Event lines
    split_epoch = warmup_epochs
    merge_epochs_abs = [warmup_epochs + e
                        for e in range(merge_every, post_epochs + 1, merge_every)
                        if e < post_epochs]

    experiments = {
        "Baseline": "baseline",
        "Baseline + Split": "split",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def add_event_lines(ax):
        ax.axvline(x=split_epoch, color="gray", linestyle="--", alpha=0.7,
                   label="Split")
        for i, me in enumerate(merge_epochs_abs):
            ax.axvline(x=me, color="red", linestyle=":", alpha=0.4,
                       label="Merge+resplit" if i == 0 else None)

    # Dead neurons layer 0
    ax = axes[0, 0]
    for label, key in experiments.items():
        mean, std = compute_mean_std(runs, key, "dead_counts", layer_idx=0)
        epochs = np.arange(len(mean))
        ax.plot(epochs, mean, label=label, marker="o", markersize=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
    add_event_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dead Neurons")
    ax.set_title(f"Dead Neurons (Layer 0) — {n_seeds} seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Dead neurons layer 1
    ax = axes[0, 1]
    for label, key in experiments.items():
        mean, std = compute_mean_std(runs, key, "dead_counts", layer_idx=1)
        epochs = np.arange(len(mean))
        ax.plot(epochs, mean, label=label, marker="o", markersize=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
    add_event_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dead Neurons")
    ax.set_title(f"Dead Neurons (Layer 1) — {n_seeds} seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Train loss
    ax = axes[1, 0]
    for label, key in experiments.items():
        mean, std = compute_mean_std(runs, key, "train_losses")
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label, marker="o", markersize=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
    add_event_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title(f"Training Loss — {n_seeds} seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test accuracy
    ax = axes[1, 1]
    for label, key in experiments.items():
        mean, std = compute_mean_std(runs, key, "test_accs")
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean * 100, label=label, marker="o", markersize=2)
        ax.fill_between(epochs, (mean - std) * 100, (mean + std) * 100, alpha=0.2)
    add_event_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Test Accuracy — {n_seeds} seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "dead_relu_recovery.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")
    plt.show()

    # Summary table
    print(f"\n{'':20s} {'Baseline':>20s} {'Split':>20s}")
    print("-" * 62)
    for layer_idx in range(2):
        bl_mean, bl_std = compute_mean_std(runs, "baseline", "dead_counts", layer_idx)
        sp_mean, sp_std = compute_mean_std(runs, "split", "dead_counts", layer_idx)
        print(f"{'Final dead L' + str(layer_idx):20s} "
              f"{bl_mean[-1]:6.1f} +/- {bl_std[-1]:<6.1f}      "
              f"{sp_mean[-1]:6.1f} +/- {sp_std[-1]:<6.1f}")

    bl_mean, bl_std = compute_mean_std(runs, "baseline", "test_accs")
    sp_mean, sp_std = compute_mean_std(runs, "split", "test_accs")
    print(f"{'Final test acc':20s} "
          f"{100*bl_mean[-1]:5.1f}% +/- {100*bl_std[-1]:<5.1f}%    "
          f"{100*sp_mean[-1]:5.1f}% +/- {100*sp_std[-1]:<5.1f}%")

    bl_mean, bl_std = compute_mean_std(runs, "baseline", "train_losses")
    sp_mean, sp_std = compute_mean_std(runs, "split", "train_losses")
    print(f"{'Final train loss':20s} "
          f"{bl_mean[-1]:6.4f} +/- {bl_std[-1]:<6.4f}    "
          f"{sp_mean[-1]:6.4f} +/- {sp_std[-1]:<6.4f}")


if __name__ == "__main__":
    main()
