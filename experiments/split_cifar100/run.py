"""Split CIFAR-100: class-incremental continual learning with ViT.

20 classes per task × 5 tasks = 100 classes.
Train on ALL seen classes, test on ALL seen. 100-class head from the start.
Data augmentation: random crop + horizontal flip (applied on-the-fly).

Usage:
    python experiments/split_cifar100/run.py
    python experiments/split_cifar100/run.py --n-layers 8 --d-model 64
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from WeightDecomp import DecomposedViT

N_TASKS = 5
CLASSES_PER_TASK = 20
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

TRAIN_AUG = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
])
NORMALIZE = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


# ========== Data ==========

def load_cifar100(data_root="data"):
    """Load raw CIFAR-100 as tensors. Augmentation applied later per-batch."""
    tr = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True)
    te = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
    # Store raw PIL images for augmentation, labels as tensors
    tr_labels = torch.tensor(tr.targets)
    te_labels = torch.tensor(te.targets)
    # Pre-normalize test data (no augmentation)
    te_data = torch.stack([NORMALIZE(te[i][0]) for i in range(len(te))])
    return tr, tr_labels, te_data, te_labels


def get_task_indices(labels, task_id):
    seen = list(range(task_id * CLASSES_PER_TASK))
    new = list(range((task_id - 1) * CLASSES_PER_TASK, task_id * CLASSES_PER_TASK))
    mask = sum(labels == c for c in seen).bool()
    indices = mask.nonzero(as_tuple=True)[0]
    return indices, seen, new


def make_batch(train_set, indices, batch_size, augment=True):
    """Sample a batch with on-the-fly augmentation."""
    chosen = indices[torch.randint(len(indices), (batch_size,))]
    imgs = []
    labels = []
    for idx in chosen:
        img, label = train_set[idx.item()]  # PIL image, int label
        if augment:
            img = TRAIN_AUG(img)
        imgs.append(NORMALIZE(img))
        labels.append(label)
    return torch.stack(imgs), torch.tensor(labels)


# ========== Utilities ==========

@torch.no_grad()
def count_dead(model, test_data, test_mask, device, max_samples=2000):
    model.eval()
    layers = model.ffn_layers()
    maxp = [torch.full((l.out_features,), -float("inf"), device=device) for l in layers]

    def hook(idx):
        def fn(m, i, o):
            maxp[idx] = torch.maximum(maxp[idx], o.amax(dim=tuple(range(o.dim()-1))))
        return fn

    hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    data = test_data[test_mask][:max_samples]
    for s in range(0, len(data), 128):
        model(data[s:s+128].to(device))
    for h in hs: h.remove()
    return sum(int((m < 0).sum()) for m in maxp), sum(m.numel() for m in maxp)


@torch.no_grad()
def evaluate(model, data, labels, device):
    model.eval()
    c, t = 0, 0
    for i in range(0, len(data), 256):
        x, y = data[i:i+256].to(device), labels[i:i+256].to(device)
        c += model(x).argmax(1).eq(y).sum().item(); t += y.size(0)
    return c / t


# ========== Training ==========

def train_task(train_model, model, train_set, train_indices, opt, crit,
               test_data, test_mask, device, args, task_id):
    train_model.train()
    n = len(train_indices)
    batches_per_epoch = n // args.batch_size
    total_loss, correct, total = 0.0, 0, 0
    dead_sum, dead_n = 0, 0

    for ep in range(args.epochs_per_task):
        if args.resplit_every > 0 and ep > 0 and ep % args.resplit_every == 0:
            model.merge_all(rerandomize_B=True)
            opt.state.clear()

        ep_loss, ep_c, ep_t = 0.0, 0, 0
        for _ in range(batches_per_epoch):
            x, y = make_batch(train_set, train_indices, args.batch_size)
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = train_model(x)
            loss = crit(out, y)
            ep_c += out.argmax(1).eq(y).sum().item()
            ep_t += y.size(0)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * y.size(0)

        total_loss += ep_loss; correct += ep_c; total += ep_t

        is_dead_ep = (ep + 1) % 10 == 0 or ep == args.epochs_per_task - 1
        if is_dead_ep:
            d, tn = count_dead(model, test_data, test_mask, device)
            dead_sum += d; dead_n += 1
            print(f"    T{task_id} ep {ep+1:2d}/{args.epochs_per_task} | "
                  f"Loss: {ep_loss/ep_t:.4f} | Acc: {100*ep_c/ep_t:.1f}% | Dead: {d}/{tn}")
        else:
            print(f"    T{task_id} ep {ep+1:2d}/{args.epochs_per_task} | "
                  f"Loss: {ep_loss/ep_t:.4f} | Acc: {100*ep_c/ep_t:.1f}%")

    avg_dead = dead_sum / dead_n if dead_n > 0 else 0
    return total_loss / total, correct / total, avg_dead, d, tn


# ========== Experiment ==========

def run_split(base_model, train_set, train_labels, test_data, test_labels,
              device, args, mode="baseline"):
    model = copy.deepcopy(base_model)
    has_factors = mode != "baseline"
    if has_factors:
        model.split_all(shared_ranks=args.ranks, local_ranks=args.ranks)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    train_model = model.compiled() if hasattr(model, 'compiled') else model

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    history = []
    t_total = 0.0

    for task_id in range(1, N_TASKS + 1):
        train_idx, seen, new = get_task_indices(train_labels, task_id)
        test_mask = sum(test_labels == c for c in seen).bool()
        old = [c for c in seen if c not in new]

        print(f"  Task {task_id}: {len(seen)} classes, {len(train_idx)} train")

        if has_factors and task_id > 1 and args.merge_every > 0:
            model.merge_all(rerandomize_B=True)
            opt.state.clear()

        t0 = time.perf_counter()
        loss, acc, avg_dead, final_dead, tn = train_task(
            train_model, model, train_set, train_idx, opt, crit,
            test_data, test_mask, device, args, task_id)
        dt = time.perf_counter() - t0
        t_total += dt

        te_seen = test_data[test_mask]
        te_y = test_labels[test_mask]
        test_acc = evaluate(train_model, te_seen, te_y, device)

        new_accs = [evaluate(train_model, test_data[test_labels == c],
                             test_labels[test_labels == c], device)
                    for c in new if (test_labels == c).sum() > 0]
        old_accs = [evaluate(train_model, test_data[test_labels == c],
                             test_labels[test_labels == c], device)
                    for c in old if (test_labels == c).sum() > 0]
        plasticity = sum(new_accs) / len(new_accs) if new_accs else 0.0
        retention = sum(old_accs) / len(old_accs) if old_accs else 1.0

        history.append({"task": task_id, "n_classes": len(seen),
                        "test_acc": test_acc, "plasticity": plasticity, "retention": retention,
                        "avg_dead": avg_dead, "final_dead": final_dead, "total": tn,
                        "time": dt, "total_time": t_total})

        print(f"  → Test: {100*test_acc:.1f}% | "
              f"New classes: {100*plasticity:.1f}% | "
              f"Old classes: {100*retention:.1f}% | "
              f"Dead: {avg_dead:.0f}/{tn} | {dt:.1f}s\n")

    return history


def main():
    p = argparse.ArgumentParser(description="Split CIFAR-100 continual learning")
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=24)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--epochs-per-task", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 2])
    p.add_argument("--merge-every", type=int, default=1)
    p.add_argument("--resplit-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"ViT: {args.n_layers}L d={args.d_model} ff={args.d_ff} patch={args.patch_size}")
    print(f"Ranks: {args.ranks}, merge/{args.merge_every}, resplit/{args.resplit_every}\n")

    train_set, train_labels, test_data, test_labels = load_cifar100()

    def make_model():
        return DecomposedViT(
            img_size=32, patch_size=args.patch_size, in_channels=3, num_classes=100,
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, d_ff=args.d_ff, skip=True).to(device)

    results = {}

    print("=== ours ===")
    torch.manual_seed(args.seed)
    results["ours"] = run_split(make_model(), train_set, train_labels,
                                test_data, test_labels, device, args, "ours")

    print("=== baseline ===")
    torch.manual_seed(args.seed)
    results["baseline"] = run_split(make_model(), train_set, train_labels,
                                    test_data, test_labels, device, args, "baseline")

    print("=" * 90)
    print(f"{'':12} | {'Task':>4} | {'Classes':>7} | {'Test':>6} | "
          f"{'New cls':>7} | {'Old cls':>7} | {'Dead':>8} | {'Time':>6}")
    print("-" * 90)
    for key in results:
        for h in results[key]:
            print(f"{key if h['task']==1 else '':12} | {h['task']:>4} | "
                  f"{h['n_classes']:>7} | {100*h['test_acc']:5.1f}% | "
                  f"{100*h['plasticity']:5.1f}% | {100*h['retention']:5.1f}% | "
                  f"{h['avg_dead']:>4.0f}/{h['total']} | {h['time']:5.1f}s")
        print(f"{'':12}   Total: {results[key][-1]['total_time']:.1f}s")
        print("-" * 90)

    Path(__file__).parent.joinpath("results.json").write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
