"""Train ViT on CIFAR-100: Hurwitz vs baseline vs BC decomposition.

Usage:
    python experiments/cifar100_hurwitz/run.py
    python experiments/cifar100_hurwitz/run.py --r 8 --K 4 --alpha 0.1
    python experiments/cifar100_hurwitz/run.py --resume checkpoints/latest
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from WeightDecomp.checkpoint import save_checkpoint, load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from WeightDecomp import DecomposedViT, HurwitzViT

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def get_loaders(batch_size, data_root="data"):
    train_t = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                         T.ToTensor(), T.Normalize(MEAN, STD)])
    test_t = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = torchvision.datasets.CIFAR100(root=data_root, train=True,
                                              transform=train_t, download=True)
    test_set = torchvision.datasets.CIFAR100(root=data_root, train=False,
                                             transform=test_t, download=True)
    return (torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(test_set, batch_size=256, num_workers=2))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    c, t = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += model(x).argmax(1).eq(y).sum().item(); t += y.size(0)
    return c / t


@torch.no_grad()
def count_dead(model, loader, device, max_batches=20):
    model.eval()
    layers = model.ffn_layers()
    maxp = [torch.full((l.out_features,), -float("inf"), device=device) for l in layers]

    def hook(idx):
        def fn(m, i, o):
            maxp[idx] = torch.maximum(maxp[idx], o.amax(dim=tuple(range(o.dim()-1))))
        return fn

    hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches: break
        model(x.to(device))
    for h in hs: h.remove()
    return sum(int((m < 0).sum()) for m in maxp), sum(m.numel() for m in maxp)


def save_ckpt(path, model, optimizer, args, epoch, history):
    save_checkpoint(path, model, optimizer,
                    metadata={"epoch": epoch, "config": vars(args), "history": history})


def load_ckpt(path, model, optimizer, device):
    meta = load_checkpoint(path, model, optimizer, device)
    return meta.get("epoch", 0), meta.get("history", [])


def train(model, args, tag="model"):
    device = torch.device(args.device)
    model = model.to(device)
    print(f"\n=== {tag} ===")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path(args.ckpt_dir or Path(__file__).parent / "checkpoints") / tag
    train_model = model.compiled() if hasattr(model, 'compiled') else model
    train_loader, test_loader = get_loaders(args.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    start_epoch = 1
    history = []
    if args.resume:
        resume_path = Path(args.resume)
        if (resume_path / "model.safetensors").exists():
            start_epoch_prev, history = load_ckpt(resume_path, model, opt, device)
            start_epoch = start_epoch_prev + 1
            # Override lr from CLI
            for pg in opt.param_groups:
                pg['lr'] = args.lr

    # Grow width if requested (after loading checkpoint)
    if (args.grow_d or args.grow_ff) and hasattr(model, 'grow_width'):
        # For BC: merge factors into W first, grow, then re-split
        if tag == "bc" and hasattr(model, 'merge_all'):
            model.merge_all(rerandomize_B=False)
        model.grow_width(
            new_d_model=args.grow_d or None,
            new_d_ff=args.grow_ff or None)
        if tag == "bc" and hasattr(model, 'split_all'):
            model.split_all(shared_ranks=args.bc_ranks, local_ranks=args.bc_ranks)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_model = model.compiled() if hasattr(model, 'compiled') else model

    t0 = time.perf_counter()

    for ep in range(start_epoch, args.epochs + 1):
        if args.resplit_every > 0 and ep > 1 and (ep - 1) % args.resplit_every == 0:
            model.merge_all()
            opt.state.clear()

        train_model.train()
        ep_loss, ep_c, ep_t = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = train_model(x)
            loss = crit(out, y)
            ep_c += out.argmax(1).eq(y).sum().item()
            ep_t += y.size(0)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * y.size(0)

        train_acc = ep_c / ep_t
        train_loss = ep_loss / ep_t

        if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
            test_acc = evaluate(train_model, test_loader, device)
            dead, total = count_dead(model, test_loader, device)
            elapsed = time.perf_counter() - t0
            history.append({"epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                            "test_acc": test_acc, "dead": dead, "total": total, "time": elapsed})
            print(f"  Ep {ep:3d}/{args.epochs} | Loss: {train_loss:.4f} | "
                  f"Train: {100*train_acc:.1f}% | Test: {100*test_acc:.1f}% | "
                  f"Dead: {dead}/{total} | {elapsed:.0f}s")
        else:
            print(f"  Ep {ep:3d}/{args.epochs} | Loss: {train_loss:.4f} | "
                  f"Train: {100*train_acc:.1f}%")

        if args.save_every > 0 and ep % args.save_every == 0:
            save_ckpt(ckpt_dir / f"ep_{ep}", model, opt, args, ep, history)
            save_ckpt(ckpt_dir / "latest", model, opt, args, ep, history)

    save_ckpt(ckpt_dir / "final", model, opt, args, args.epochs, history)
    return history


def main():
    p = argparse.ArgumentParser(description="CIFAR-100: Hurwitz vs BC vs baseline")
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=48)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    # BC args
    p.add_argument("--bc-ranks", nargs="+", type=int, default=[4])
    # Hurwitz args
    p.add_argument("--r", type=int, default=4)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    # Training
    p.add_argument("--resplit-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--grow-d", type=int, default=0,
                    help="Grow d_model to this size on resume (0 = no growth)")
    p.add_argument("--grow-ff", type=int, default=0,
                    help="Grow d_ff to this size on resume (0 = no growth)")
    p.add_argument("--methods", nargs="+", default=["baseline"],# ["hurwitz", "bc", "baseline"],
                    help="Which methods to run")
    args = p.parse_args()

    print(f"Device: {args.device}")
    print(f"ViT: {args.n_layers}L d={args.d_model} ff={args.d_ff} patch={args.patch_size}")
    print(f"Hurwitz: r={args.r} K={args.K} alpha={args.alpha}")
    print(f"BC: ranks={args.bc_ranks}")
    print(f"Methods: {args.methods}")

    results = {}

    vit_kw = dict(img_size=32, patch_size=args.patch_size, in_channels=3,
                  num_classes=100, d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, d_ff=args.d_ff)

    if "hurwitz" in args.methods:
        torch.manual_seed(args.seed)
        model = HurwitzViT(**vit_kw, r=args.r, K=args.K, alpha=args.alpha)
        results["hurwitz"] = train(model, args, "hurwitz")

    if "bc" in args.methods:
        torch.manual_seed(args.seed)
        model = DecomposedViT(**vit_kw, skip=True, alpha=args.alpha)
        model.split_all(shared_ranks=args.bc_ranks, local_ranks=args.bc_ranks)
        results["bc"] = train(model, args, "bc")

    if "baseline" in args.methods:
        torch.manual_seed(args.seed)
        model = DecomposedViT(**vit_kw, skip=True, alpha=args.alpha)
        results["baseline"] = train(model, args, "baseline")

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Method':<12} | {'Ep':>4} | {'Train':>6} | {'Test':>6} | {'Dead':>8} | {'Time':>6}")
    print("-" * 80)
    for key in results:
        h = results[key][-1]
        print(f"{key:<12} | {h['epoch']:>4} | {100*h['train_acc']:5.1f}% | "
              f"{100*h['test_acc']:5.1f}% | {h['dead']:>4}/{h['total']} | "
              f"{h['time']:.0f}s")

    Path(__file__).parent.joinpath("results.json").write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
