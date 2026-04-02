"""Split MNIST: class-incremental continual learning.

Train on ALL seen classes, test on ALL seen. 10-class head from the start.

Usage:
    python experiments/split_mnist/run.py --model mlp
    python experiments/split_mnist/run.py --model vit --device cuda
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
from WeightDecomp import DecomposedMLP, DecomposedViT


# ========== Data ==========

class SplitMNIST:
    TASK_CLASSES = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
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

    def get_task_data(self, task_id, model_type="mlp"):
        new = self.TASK_CLASSES[task_id - 1]
        seen = [c for t in range(task_id) for c in self.TASK_CLASSES[t]]
        tr_mask = sum(self.train_labels == c for c in seen).bool()
        te_mask = sum(self.test_labels == c for c in seen).bool()
        tr_x = self.train_data[tr_mask]
        te_x = self.test_data[te_mask]
        if model_type == "vit":
            tr_x = tr_x.unsqueeze(1)  # (N, 1, 28, 28)
            te_x = te_x.unsqueeze(1)
        else:
            tr_x = tr_x.view(len(tr_x), -1)
            te_x = te_x.view(len(te_x), -1)
        return tr_x, self.train_labels[tr_mask], te_x, self.test_labels[te_mask], seen, new


# ========== Utilities ==========

@torch.no_grad()
def count_dead(model, data, device, max_samples=3000):
    model.eval()
    layers = model.ffn_layers() if hasattr(model, 'ffn_layers') else model.decomposed_layers()[:-1]
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

def train_task(train_model, model, tr_x, tr_y, opt, crit, device, args, task_id):
    train_model.train()
    n = len(tr_x)
    total_loss, correct, total = 0.0, 0, 0
    dead_sum, dead_n = 0, 0

    for ep in range(args.epochs_per_task):
        if args.resplit_every > 0 and ep > 0 and ep % args.resplit_every == 0:
            model.merge_all(rerandomize_B=True)
            opt.state.clear()

        ep_loss, ep_c, ep_t = 0.0, 0, 0
        for i in range(0, n, args.batch_size):
            idx = torch.randint(n, (min(args.batch_size, n - i),))
            x, y = tr_x[idx].to(device), tr_y[idx].to(device)
            opt.zero_grad()
            out = train_model(x)
            loss = crit(out, y)
            ep_c += out.argmax(1).eq(y).sum().item()
            ep_t += y.size(0)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * y.size(0)

        total_loss += ep_loss; correct += ep_c; total += ep_t

        # Count dead only every 5 epochs to save time
        if (ep + 1) % 5 == 0 or ep == args.epochs_per_task - 1:
            d, tn = count_dead(model, tr_x, device)
            dead_sum += d; dead_n += 1
            print(f"    T{task_id} ep {ep+1:2d}/{args.epochs_per_task} | "
                  f"Loss: {ep_loss/ep_t:.4f} | Acc: {100*ep_c/ep_t:.1f}% | Dead: {d}/{tn}")

    avg_dead = dead_sum / dead_n if dead_n > 0 else 0
    return total_loss / total, correct / total, avg_dead, d, tn


# ========== Experiment ==========

def build_model(args):
    if args.model == "vit":
        return DecomposedViT(img_size=28, patch_size=7, in_channels=1, num_classes=10,
                             d_model=args.d_model, n_heads=args.n_heads,
                             n_layers=args.n_layers, d_ff=args.d_ff, skip=True)
    return DecomposedMLP([784] + [args.hidden_size] * args.n_hidden + [10])


def run_split(base_model, dataset, device, args, mode="baseline"):
    model = copy.deepcopy(base_model)
    has_factors = mode != "baseline"
    if has_factors:
        if args.model == "vit":
            model.split_all(shared_ranks=args.ranks, local_ranks=args.ranks)
        else:
            model.split_all(args.ranks)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    train_model = model.compiled() if hasattr(model, 'compiled') else \
        torch.compile(model, mode="reduce-overhead")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    history = []
    t_total = 0.0

    for task_id in range(1, 6):
        tr_x, tr_y, te_x, te_y, seen, new = dataset.get_task_data(task_id, args.model)
        old = [c for c in seen if c not in new]
        print(f"  Task {task_id}: {seen}, {len(tr_x)} train")

        if has_factors and task_id > 1 and args.merge_every > 0:
            model.merge_all(rerandomize_B=True)
            opt.state.clear()

        t0 = time.perf_counter()
        loss, acc, avg_dead, final_dead, tn = train_task(
            train_model, model, tr_x, tr_y, opt, crit, device, args, task_id)
        dt = time.perf_counter() - t0
        t_total += dt

        ca = {c: evaluate(train_model, te_x[te_y == c], te_y[te_y == c], device)
              for c in seen if (te_y == c).sum() > 0}
        plast = sum(ca[c] for c in new if c in ca) / len(new)
        retain = sum(ca[c] for c in old if c in ca) / len(old) if old else 1.0
        test_acc = evaluate(train_model, te_x, te_y, device)

        history.append({"task": task_id, "seen": seen, "new": new,
                        "test_acc": test_acc, "plasticity": plast, "retention": retain,
                        "class_accs": ca, "avg_dead": avg_dead, "final_dead": final_dead,
                        "total": tn, "time": dt, "total_time": t_total})

        cs = " ".join(f"{c}:{100*ca[c]:.0f}%" for c in sorted(ca))
        print(f"  → {100*test_acc:.1f}% | P:{100*plast:.1f}% R:{100*retain:.1f}% | "
              f"Dead:{avg_dead:.0f}/{tn} | {dt:.1f}s\n    {cs}\n")

    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "vit"], default="mlp")
    p.add_argument("--hidden-size", type=int, default=100)
    p.add_argument("--n-hidden", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--epochs-per-task", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ranks", nargs="+", type=int, default=[4, 8])
    p.add_argument("--merge-every", type=int, default=1)
    p.add_argument("--resplit-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}, Model: {args.model}")
    print(f"Ranks: {args.ranks}, merge/{args.merge_every}, resplit/{args.resplit_every}\n")

    dataset = SplitMNIST.from_torchvision()
    results = {}

    print("=== ours ===")
    torch.manual_seed(args.seed)
    results["ours"] = run_split(build_model(args).to(device), dataset, device, args, "ours")

    print("=== baseline ===")
    torch.manual_seed(args.seed)
    results["baseline"] = run_split(build_model(args).to(device), dataset, device, args, "baseline")

    print("=" * 80)
    print(f"{'':12} | {'Task':>4} | {'Acc':>6} | {'Plast':>6} | {'Retain':>6} | {'Dead':>8} | {'Time':>6}")
    print("-" * 80)
    for key in results:
        for h in results[key]:
            print(f"{key if h['task']==1 else '':12} | {h['task']:>4} | "
                  f"{100*h['test_acc']:5.1f}% | {100*h['plasticity']:5.1f}% | "
                  f"{100*h['retention']:5.1f}% | {h['avg_dead']:>4.0f}/{h['total']} | "
                  f"{h['time']:5.1f}s")
        print(f"{'':12}   Total: {results[key][-1]['total_time']:.1f}s")
        print("-" * 80)

    Path(__file__).parent.joinpath("results.json").write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
