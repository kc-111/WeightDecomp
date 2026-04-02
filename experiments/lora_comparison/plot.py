"""Plot LoRA vs ReLoRA vs W+BC comparison results.

Usage:
    python experiments/lora_comparison/plot.py
    python experiments/lora_comparison/plot.py --save fig.png
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

    # Group by transform
    transforms_used = config["transforms"]
    modes = ["baseline", "lora", "relora", "ours"]
    mode_colors = {"baseline": "gray", "lora": "red", "relora": "orange", "ours": "blue"}
    mode_styles = {"baseline": "--", "lora": "-.", "relora": ":", "ours": "-"}

    n_transforms = len(transforms_used)
    fig, axes = plt.subplots(n_transforms, 3, figsize=(15, 4 * n_transforms),
                             squeeze=False)
    fig.suptitle(
        f"LoRA vs W+BC — {config['layer_sizes']}, ranks={config['ranks']}",
        fontsize=13, y=1.02,
    )

    for row, bt in enumerate(transforms_used):
        for col, (metric, ylabel, title) in enumerate([
            ("dead", "Dead Neurons", f"Dead Neuron Revival (B={bt})"),
            ("loss", "Train Loss", f"Training Loss (B={bt})"),
            ("test_acc", "Test Accuracy (%)", f"Test Accuracy (B={bt})"),
        ]):
            ax = axes[row, col]
            for mode in modes:
                key = f"{bt}_{mode}"
                if key not in results:
                    continue
                hist = results[key]
                epochs = [h["epoch"] for h in hist]
                if metric == "test_acc":
                    values = [100 * h[metric] for h in hist]
                else:
                    values = [h[metric] for h in hist]
                ax.plot(epochs, values, label=mode, marker="o", markersize=3,
                        color=mode_colors[mode], linestyle=mode_styles[mode])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
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
