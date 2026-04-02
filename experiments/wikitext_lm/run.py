"""Train a small decoder-only LM on WikiText-103.

Reports per-iteration: train loss, val perplexity, dead neurons.
Supports checkpoint save/resume with safetensors.

Usage:
    python experiments/wikitext_lm/run.py                                    # baseline
    python experiments/wikitext_lm/run.py --ranks 4 8 --resplit-every 2000   # ours
    python experiments/wikitext_lm/run.py --resume checkpoints/latest        # resume
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken
from datasets import load_from_disk
from WeightDecomp.checkpoint import save_checkpoint as _save_ckpt, load_checkpoint as _load_ckpt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from WeightDecomp import DecomposedLM


# ========== Data ==========

def tokenize_dataset(dataset_path, split="train", cache_dir=None):
    from safetensors.torch import save_file, load_file
    cache_file = Path(cache_dir or dataset_path) / f"{split}_tokens.safetensors"
    if cache_file.exists():
        print(f"Loading cached tokens from {cache_file}")
        return load_file(str(cache_file))["tokens"]

    print(f"Tokenizing {split}...")
    ds = load_from_disk(dataset_path)[split]
    enc = tiktoken.get_encoding("gpt2")
    all_tokens = []
    for i, example in enumerate(ds):
        text = example["text"]
        if text.strip():
            all_tokens.extend(enc.encode(text, allowed_special=set()))
        if (i + 1) % 200000 == 0:
            print(f"  {i+1}/{len(ds)} ({len(all_tokens):,} tokens)")
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"  {split}: {len(tokens):,} tokens")
    save_file({"tokens": tokens}, str(cache_file))
    return tokens


def make_chunks(tokens, seq_len):
    n = len(tokens) // (seq_len + 1)
    return tokens[:n * (seq_len + 1)].view(n, seq_len + 1)


# ========== Checkpointing ==========

def save_checkpoint(path, model, optimizer, args, global_iter, epoch, history):
    _save_ckpt(path, model, optimizer,
               metadata={"global_iter": global_iter, "epoch": epoch,
                          "config": vars(args), "history": history})


def load_checkpoint(path, model, optimizer, device):
    meta = _load_ckpt(path, model, optimizer, device)
    return meta.get("global_iter", 0), meta.get("epoch", 0), meta.get("history", [])


# ========== Utilities ==========

@torch.no_grad()
def evaluate_loss(model, chunks, device, max_batches=50, batch_size=32):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    perm = torch.randperm(len(chunks))
    for i in range(0, min(len(chunks), max_batches * batch_size), batch_size):
        batch = chunks[perm[i:i+batch_size]].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / total_tokens


@torch.no_grad()
def count_dead(model, chunks, device, max_batches=10, batch_size=16):
    model.eval()
    layers = model.ffn_layers()
    maxp = [torch.full((l.out_features,), -float("inf"), device=device) for l in layers]

    def hook(idx):
        def fn(m, i, o):
            maxp[idx] = torch.maximum(maxp[idx], o.amax(dim=tuple(range(o.dim()-1))))
        return fn

    hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    for bi in range(min(max_batches, len(chunks) // batch_size)):
        model(chunks[bi*batch_size:(bi+1)*batch_size, :-1].to(device))
    for h in hs: h.remove()
    return sum(int((m < 0).sum()) for m in maxp), sum(m.numel() for m in maxp)


# ========== Training ==========

def main():
    p = argparse.ArgumentParser(description="Train LM on WikiText-103")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-iters", type=int, default=50000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=5000,
                    help="Save checkpoint every N iters (0 = only at end)")
    p.add_argument("--ranks", nargs="+", type=int, default=[])
    p.add_argument("--resplit-every", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-path", type=str, default="data/wikitext-103")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint directory to resume from")
    p.add_argument("--ckpt-dir", type=str, default=None,
                    help="Checkpoint directory (default: experiments/wikitext_lm/checkpoints)")
    p.add_argument("--grow-ff", type=int, default=0,
                    help="Grow FFN hidden width to this size on resume (0 = no growth)")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(__file__).parent / "checkpoints"

    print(f"Device: {device}")
    print(f"LM: {args.n_layers}L d={args.d_model} ff={args.d_ff} seq={args.seq_len}")
    if args.ranks:
        print(f"Ranks: {args.ranks}, resplit/{args.resplit_every}")
    else:
        print("No decomposition (baseline)")
    print()

    # Data
    train_tokens = tokenize_dataset(args.data_path, "train")
    val_tokens = tokenize_dataset(args.data_path, "validation")
    train_chunks = make_chunks(train_tokens, args.seq_len)
    val_chunks = make_chunks(val_tokens, args.seq_len)
    print(f"Train: {len(train_chunks):,} chunks | Val: {len(val_chunks):,} chunks\n")

    # Model
    model = DecomposedLM(
        vocab_size=50257, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_seq_len=args.seq_len,
        dropout=args.dropout).to(device)

    if args.ranks:
        model.split_all(shared_ranks=args.ranks, local_ranks=args.ranks)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume
    global_iter = 0
    epoch = 0
    history = []
    if args.resume:
        global_iter, epoch, history = load_checkpoint(args.resume, model, opt, device)
        for pg in opt.param_groups:
            pg['lr'] = args.lr

    # Grow width if requested (after loading checkpoint)
    if args.grow_ff:
        from WeightDecomp.vit import SharedFactorLinear
        old_ff = model.d_ff
        if args.grow_ff > old_ff:
            for block in model.blocks:
                block.ffn.fc1.grow(new_out=args.grow_ff)
                block.ffn.fc2.grow(new_in=args.grow_ff)
            model.d_ff = args.grow_ff
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            print(f"  Grew FFN width {old_ff} → {args.grow_ff}, "
                  f"new params: {sum(p.numel() for p in model.parameters()):,}")

    train_model = model.compiled() if hasattr(model, 'compiled') else model
    crit = nn.CrossEntropyLoss()
    t0 = time.perf_counter()

    while global_iter < args.max_iters:
        epoch += 1
        perm = torch.randperm(len(train_chunks))
        for i in range(0, len(train_chunks), args.batch_size):
            if global_iter >= args.max_iters:
                break

            if args.resplit_every > 0 and args.ranks and global_iter > 0 \
                    and global_iter % args.resplit_every == 0:
                model.merge_all(rerandomize_B=True)
                opt.state.clear()
                print(f"  [Merge+resplit at iter {global_iter}]")

            idx = perm[i:i+args.batch_size]
            batch = train_chunks[idx].to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            opt.zero_grad()
            train_model.train()
            logits = train_model(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            global_iter += 1

            # Eval
            if global_iter % args.eval_every == 0 or global_iter == 1:
                val_loss = evaluate_loss(train_model, val_chunks, device)
                dead, total = count_dead(model, train_chunks, device)
                elapsed = time.perf_counter() - t0
                ppl = math.exp(min(val_loss, 20))
                history.append({"iter": global_iter, "epoch": epoch,
                                "train_loss": loss.item(), "val_loss": val_loss,
                                "val_ppl": ppl, "dead": dead, "total": total,
                                "time": elapsed})
                print(f"iter {global_iter:6d} | ep {epoch} | "
                      f"train {loss.item():.3f} | val {val_loss:.3f} | "
                      f"ppl {ppl:.1f} | dead {dead}/{total} | {elapsed:.0f}s")

            # Save checkpoint
            if args.save_every > 0 and global_iter % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"iter_{global_iter}",
                                model, opt, args, global_iter, epoch, history)
                save_checkpoint(ckpt_dir / "latest",
                                model, opt, args, global_iter, epoch, history)

    # Final save
    save_checkpoint(ckpt_dir / "final", model, opt, args, global_iter, epoch, history)

    # Results JSON
    Path(__file__).parent.joinpath("results.json").write_text(
        json.dumps({"config": vars(args), "results": history}, indent=2, default=str))


if __name__ == "__main__":
    main()
