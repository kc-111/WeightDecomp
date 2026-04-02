"""Plot Split MNIST results.

Usage:
    python experiments/split_mnist/plot.py
    python experiments/split_mnist/plot.py --save fig.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        f"Split MNIST — {config['n_hidden']}×{config['hidden_size']}, "
        f"ranks={config['ranks']}",
        fontsize=12,
    )

    task_labels = ["0-1", "0-3", "0-5", "0-7", "0-9"]
    x = np.arange(5)

    colors = {"baseline": "gray"}
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    ours_keys = [k for k in results if k != "baseline"]
    for i, k in enumerate(ours_keys):
        colors[k] = palette[i % len(palette)]

    # 1. Test accuracy per task
    ax = axes[0]
    for key in results:
        accs = [100 * h["test_acc"] for h in results[key]]
        style = ":" if key == "baseline" else "-"
        ax.plot(x, accs, label=key, marker="o", color=colors.get(key, "black"),
                linestyle=style, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_xlabel("Classes seen")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy on All Seen Classes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Dead neurons
    ax = axes[1]
    for key in results:
        dead = [h["dead"] for h in results[key]]
        style = ":" if key == "baseline" else "-"
        ax.plot(x, dead, label=key, marker="s", color=colors.get(key, "black"),
                linestyle=style, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_xlabel("Classes seen")
    ax.set_ylabel("Dead Neurons")
    ax.set_title("Dead Neuron Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Per-class accuracy at final task
    ax = axes[2]
    width = 0.8 / len(results)
    for j, key in enumerate(results):
        final = results[key][-1]
        class_accs = final["class_accs"]
        classes = sorted(int(c) for c in class_accs.keys())
        accs = [100 * class_accs[str(c)] for c in classes]
        offset = (j - len(results) / 2 + 0.5) * width
        ax.bar([c + offset for c in classes], accs, width,
               label=key, color=colors.get(key, "black"), alpha=0.8)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy (after all 5 tasks)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
