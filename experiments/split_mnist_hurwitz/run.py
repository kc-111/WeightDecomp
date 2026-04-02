"""Split MNIST with Hurwitz decomposition vs baseline vs plain BC.

Compares:
  1. baseline:  standard MLP
  2. bc:        W + BC decomposition (DecomposedMLP)
  3. hurwitz:   W + A @ H @ D where H is Hurwitz (HurwitzMLP)

Usage:
    python experiments/split_mnist_hurwitz/run.py
    python experiments/split_mnist_hurwitz/run.py --r 8 --K 4 --epochs-per-task 100
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from WeightDecomp import DecomposedMLP, HurwitzMLP


# ========== Data ==========

class SplitMNIST:
    TASK_CLASSES = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data.view(-1, 784)
        self.train_labels = train_labels
        self.test_data = test_data.view(-1, 784)
        self.test_labels = test_labels

    @staticmethod
    def from_torchvision(data_root="data"):
        import torchvision
        import torchvision.transforms as T
        t = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        tr = torchvision.datasets.MNIST(root=data_root, train=True, transform=t, download=True)
        te = torchvision.datasets.MNIST(root=data_root, train=False, transform=t, download=True)
        trd = (tr.data.float() / 255.0 - 0.1307) / 0.3081
        ted = (te.data.float() / 255.0 - 0.1307) / 0.3081
        return SplitMNIST(trd, tr.targets, ted, te.targets)

    def get_task_data(self, task_id):
        new = self.TASK_CLASSES[task_id - 1]
        seen = [c for t in range(task_id) for c in self.TASK_CLASSES[t]]
        tr_mask = sum(self.train_labels == c for c in seen).bool()
        te_mask = sum(self.test_labels == c for c in seen).bool()
        return (self.train_data[tr_mask], self.train_labels[tr_mask],
                self.test_data[te_mask], self.test_labels[te_mask], seen, new)


# ========== Utilities ==========

@torch.no_grad()
def count_dead(model, data, device, max_samples=3000):
    model.eval()
    layers = model.decomposed_layers()[:-1]
    maxp = [torch.full((l.out_features,), -float("inf"), device=device) for l in layers]

    def hook(idx):
        def fn(m, i, o):
            maxp[idx] = torch.maximum(maxp[idx], o.amax(dim=tuple(range(o.dim()-1))))
        return fn

    hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    for s in range(0, min(len(data), max_samples), 256):
        model(data[s:s+256].to(device))
    for h in hs: h.remove()
    return sum(int((m < 0).sum()) for m in maxp), sum(m.numel() for m in maxp)


@torch.no_grad()
def evaluate(model, data, labels, device):
    model.eval()
    c, t = 0, 0
    for i in range(0, len(data), 512):
        x, y = data[i:i+512].to(device), labels[i:i+512].to(device)
        c += model(x).argmax(1).eq(y).sum().item(); t += y.size(0)
    return c / t


# ========== Training ==========

def train_task(model, tr_x, tr_y, opt, crit, device, args, task_id):
    model.train()
    n = len(tr_x)
    total_loss, correct, total = 0.0, 0, 0
    dead_sum, dead_n = 0, 0

    for ep in range(args.epochs_per_task):
        if args.resplit_every > 0 and ep > 0 and ep % args.resplit_every == 0:
            model.merge_all()
            opt.state.clear()

        ep_loss, ep_c, ep_t = 0.0, 0, 0
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i+args.batch_size]
            x, y = tr_x[idx].to(device), tr_y[idx].to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            ep_c += out.argmax(1).eq(y).sum().item()
            ep_t += y.size(0)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * y.size(0)

        total_loss += ep_loss; correct += ep_c; total += ep_t

        if (ep + 1) % 5 == 0 or ep == args.epochs_per_task - 1:
            d, tn = count_dead(model, tr_x, device)
            dead_sum += d; dead_n += 1
            print(f"    T{task_id} ep {ep+1:2d}/{args.epochs_per_task} | "
                  f"Loss: {ep_loss/ep_t:.4f} | Acc: {100*ep_c/ep_t:.1f}% | Dead: {d}/{tn}")

    avg_dead = dead_sum / dead_n if dead_n > 0 else 0
    return total_loss / total, correct / total, avg_dead, d, tn


# ========== Experiment ==========

def run_split(model, dataset, device, args, name):
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    history = []
    t_total = 0.0

    for task_id in range(1, 6):
        tr_x, tr_y, te_x, te_y, seen, new = dataset.get_task_data(task_id)
        old = [c for c in seen if c not in new]
        print(f"  Task {task_id}: classes {seen}, {len(tr_x)} train")

        if task_id > 1 and args.merge_every > 0:
            model.merge_all()
            opt.state.clear()

        t0 = time.perf_counter()
        loss, acc, avg_dead, final_dead, tn = train_task(
            model, tr_x, tr_y, opt, crit, device, args, task_id)
        dt = time.perf_counter() - t0
        t_total += dt

        ca = {c: evaluate(model, te_x[te_y == c], te_y[te_y == c], device)
              for c in seen if (te_y == c).sum() > 0}
        plast = sum(ca[c] for c in new if c in ca) / len(new)
        retain = sum(ca[c] for c in old if c in ca) / len(old) if old else 1.0
        test_acc = evaluate(model, te_x, te_y, device)

        history.append({"task": task_id, "test_acc": test_acc,
                        "plasticity": plast, "retention": retain,
                        "avg_dead": avg_dead, "total": tn,
                        "time": dt, "total_time": t_total})

        cs = " ".join(f"{c}:{100*ca[c]:.0f}%" for c in sorted(ca))
        print(f"  → {100*test_acc:.1f}% | New: {100*plast:.1f}% Old: {100*retain:.1f}% | "
              f"Dead: {avg_dead:.0f}/{tn} | {dt:.1f}s\n    {cs}\n")

    return history


def main():
    p = argparse.ArgumentParser(description="Split MNIST: Hurwitz vs BC vs baseline")
    p.add_argument("--hidden-size", type=int, default=100)
    p.add_argument("--n-hidden", type=int, default=3)
    p.add_argument("--epochs-per-task", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    # BC args
    p.add_argument("--bc-ranks", nargs="+", type=int, default=[4, 8])
    # Hurwitz args
    p.add_argument("--r", type=int, default=4, help="Rank per Hurwitz component")
    p.add_argument("--K", type=int, default=4, help="Number of Hurwitz components")
    p.add_argument("--alpha", type=float, default=0.1, help="Scaling on base W")
    # Merge
    p.add_argument("--merge-every", type=int, default=1)
    p.add_argument("--resplit-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    sizes = [784] + [args.hidden_size] * args.n_hidden + [10]
    print(f"Device: {device}")
    print(f"MLP: {sizes}")
    print(f"BC ranks: {args.bc_ranks} | Hurwitz r={args.r} K={args.K} alpha={args.alpha}")
    print(f"Merge/{args.merge_every}, resplit/{args.resplit_every}\n")

    dataset = SplitMNIST.from_torchvision()
    results = {}

    # BC (with same alpha as Hurwitz for fair comparison)
    print("=== bc ===")
    torch.manual_seed(args.seed)
    bcmodel = DecomposedMLP(sizes, alpha=args.alpha).to(device)
    bcmodel.split_all(args.bc_ranks)
    results["bc"] = run_split(bcmodel, dataset, device, args, "bc")

    # Hurwitz
    print("=== hurwitz ===")
    torch.manual_seed(args.seed)
    hmodel = HurwitzMLP(sizes, r=args.r, K=args.K, alpha=args.alpha).to(device)
    results["hurwitz"] = run_split(hmodel, dataset, device, args, "hurwitz")

    # Baseline (same alpha for fair comparison)
    print("=== baseline ===")
    torch.manual_seed(args.seed)
    basemodel = DecomposedMLP(sizes, alpha=args.alpha).to(device)
    results["baseline"] = run_split(basemodel, dataset, device, args, "baseline")

    # Summary
    print("=" * 90)
    print(f"{'Method':<12} | {'Task':>4} | {'Acc':>6} | {'New cls':>7} | {'Old cls':>7} | {'Dead':>8} | {'Time':>6}")
    print("-" * 90)
    for key in results:
        for h in results[key]:
            print(f"{key if h['task']==1 else '':12} | {h['task']:>4} | "
                  f"{100*h['test_acc']:5.1f}% | {100*h['plasticity']:5.1f}% | "
                  f"{100*h['retention']:5.1f}% | {h['avg_dead']:>4.0f}/{h['total']} | "
                  f"{h['time']:5.1f}s")
        print(f"{'':12}   Total: {results[key][-1]['total_time']:.1f}s")
        print("-" * 90)

    Path(__file__).parent.joinpath("results.json").write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
