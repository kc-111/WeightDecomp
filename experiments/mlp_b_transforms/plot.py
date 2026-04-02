"""Plot results from the MLP B-transform experiment.

Usage:
    python experiments/mlp_b_transforms/plot.py                    # default
    python experiments/mlp_b_transforms/plot.py --results path.json
    python experiments/mlp_b_transforms/plot.py --save fig.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default=str(Path(__file__).parent / "results.json"))
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of showing")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    config = data["config"]
    results = data["results"]
    names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"MLP B-Transform Comparison — {config['layer_sizes']}, "
        f"ranks={config['ranks']}, kill={config['kill_frac']}",
        fontsize=12,
    )

    # 1. Dead neurons over epochs
    ax = axes[0]
    for name in names:
        hist = results[name]
        epochs = [h["epoch"] for h in hist]
        dead = [h["dead"] for h in hist]
        ax.plot(epochs, dead, marker="o", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Dead Neurons")
    ax.set_title("Dead Neuron Revival")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Training loss
    ax = axes[1]
    for name in names:
        hist = results[name]
        epochs = [h["epoch"] for h in hist]
        loss = [h["loss"] for h in hist]
        ax.plot(epochs, loss, marker="o", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Test accuracy
    ax = axes[2]
    for name in names:
        hist = results[name]
        epochs = [h["epoch"] for h in hist]
        acc = [100 * h["test_acc"] for h in hist]
        ax.plot(epochs, acc, marker="o", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
