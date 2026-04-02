"""Plot shared-B ABC vs local BC comparison results.

Usage:
    python experiments/shared_abc/plot.py
    python experiments/shared_abc/plot.py --save fig.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default=str(Path(__file__).parent / "results.json"))
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    config = data["config"]
    results = data["results"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Shared ABC vs Local BC — {config['layer_sizes']}, ranks={config['ranks']}",
        fontsize=12,
    )

    # Color: baseline=gray, local=warm colors, shared=cool colors
    colors = {}
    warm = ["#e74c3c", "#e67e22", "#f1c40f", "#e91e63", "#ff5722"]
    cool = ["#3498db", "#2ecc71", "#9b59b6", "#1abc9c", "#0097a7"]
    colors["baseline"] = "gray"
    local_keys = sorted(k for k in results if k.startswith("local_"))
    shared_keys = sorted(k for k in results if k.startswith("shared_"))
    for i, k in enumerate(local_keys):
        colors[k] = warm[i % len(warm)]
    for i, k in enumerate(shared_keys):
        colors[k] = cool[i % len(cool)]

    for col, (metric, ylabel, title) in enumerate([
        ("dead", "Dead Neurons", "Dead Neuron Revival"),
        ("loss", "Train Loss", "Training Loss"),
        ("test_acc", "Test Accuracy (%)", "Test Accuracy"),
    ]):
        ax = axes[col]
        for key in results:
            hist = results[key]
            epochs = [h["epoch"] for h in hist]
            if metric == "test_acc":
                values = [100 * h[metric] for h in hist]
            else:
                values = [h[metric] for h in hist]
            style = "--" if key.startswith("local_") else "-"
            if key == "baseline":
                style = ":"
            ax.plot(epochs, values, label=key, marker="o", markersize=3,
                    color=colors.get(key, "black"), linestyle=style)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
