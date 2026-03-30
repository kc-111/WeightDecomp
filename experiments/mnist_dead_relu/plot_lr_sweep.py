"""Plot learning rate sensitivity results across multiple seeds.

Reads the same JSON format as plot.py, but organized under lr_*/seed_*/.
One figure per LR, each showing baseline vs split on the same axes.

Usage:
    python plot_lr_sweep.py
    python plot_lr_sweep.py --results-dir results_lr_sweep
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot import load_all_seeds, compute_mean_std


def load_lr_sweep(results_dir):
    """Load results for all LRs. Each lr_* subdirectory has the same
    structure as the default results dir (seed_*/baseline.json, split.json).

    Returns:
        dict mapping lr_label -> list of runs (same format as load_all_seeds)
    """
    results_dir = Path(results_dir)
    data = {}
    for lr_dir in sorted(results_dir.glob("lr_*")):
        lr_label = lr_dir.name.replace("lr_", "")
        runs = load_all_seeds(lr_dir)
        if runs:
            data[lr_label] = runs
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot LR sweep results")
    parser.add_argument("--results-dir", type=str, default="results_lr_sweep")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for figures (default: results-dir/figures)")
    args = parser.parse_args()

    data = load_lr_sweep(args.results_dir)
    if not data:
        print(f"No results found in {args.results_dir}/")
        return

    lr_labels = list(data.keys())
    n_seeds = len(next(iter(data.values())))
    config = next(iter(data.values()))[0]["config"]
    warmup_epochs = config.get("warmup_epochs", 2)
    post_epochs = config.get("post_epochs", 40)
    merge_every = config.get("merge_resplit_every", 10)

    output_dir = Path(args.output) if args.output else Path(args.results_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(lr_labels)} learning rates, {n_seeds} seeds each")
    print(f"LRs: {lr_labels}")

    # Event lines
    split_epoch = warmup_epochs
    merge_epochs_abs = [warmup_epochs + e
                        for e in range(merge_every, post_epochs + 1, merge_every)
                        if e < post_epochs]

    experiments = {
        "Baseline": "baseline",
        "Baseline + Split": "split",
    }

    def add_event_lines(ax):
        ax.axvline(x=split_epoch, color="gray", linestyle="--", alpha=0.5,
                   label="Split")
        for i, me in enumerate(merge_epochs_abs):
            ax.axvline(x=me, color="gray", linestyle=":", alpha=0.3,
                       label="Merge+resplit" if i == 0 else None)

    # --- One figure per LR, baseline vs split on same axes ---
    for lr_label in lr_labels:
        runs = data[lr_label]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

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

        plt.suptitle(f"lr = {lr_label}", fontsize=13)
        plt.tight_layout()
        fig_path = output_dir / f"lr_{lr_label}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved {fig_path}")
        plt.show()

    # --- Summary table ---
    print(f"\n{'LR':>10s} | {'BL Dead L0':>15s} | {'SP Dead L0':>15s} | "
          f"{'BL Acc':>15s} | {'SP Acc':>15s}")
    print("-" * 80)
    for lr_label in lr_labels:
        runs = data[lr_label]
        bl_d0_m, bl_d0_s = compute_mean_std(runs, "baseline", "dead_counts", 0)
        sp_d0_m, sp_d0_s = compute_mean_std(runs, "split", "dead_counts", 0)
        bl_a_m, bl_a_s = compute_mean_std(runs, "baseline", "test_accs")
        sp_a_m, sp_a_s = compute_mean_std(runs, "split", "test_accs")
        print(f"{lr_label:>10s} | "
              f"{bl_d0_m[-1]:5.1f} +/- {bl_d0_s[-1]:<4.1f} | "
              f"{sp_d0_m[-1]:5.1f} +/- {sp_d0_s[-1]:<4.1f} | "
              f"{bl_a_m[-1]*100:5.1f}% +/- {bl_a_s[-1]*100:<4.1f}% | "
              f"{sp_a_m[-1]*100:5.1f}% +/- {sp_a_s[-1]*100:<4.1f}%")


if __name__ == "__main__":
    main()
