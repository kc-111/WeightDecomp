"""Train a DecomposedMLP on MNIST with optional periodic merge."""

import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from .decomposed_linear import DecomposedLinear
from .mlp import DecomposedMLP


def reset_factor_optimizer_state(optimizer: torch.optim.Optimizer, model: nn.Module):
    """Reset optimizer state for all factor parameters after merge.

    Delegates to model.reset_factor_optimizer_state() if available (DecomposedViT).
    Falls back to scanning for DecomposedLinear modules (DecomposedMLP).
    """
    if hasattr(model, 'reset_factor_optimizer_state'):
        model.reset_factor_optimizer_state(optimizer)
    else:
        # Fallback for DecomposedMLP and other simple models
        for module in model.modules():
            if isinstance(module, DecomposedLinear):
                for param in list(module.Bs.parameters()) + list(module.Cs.parameters()):
                    if param in optimizer.state:
                        del optimizer.state[param]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train DecomposedMLP on MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--ranks", nargs="+", type=int, default=[])
    parser.add_argument("--merge-every", type=int, default=0, help="Merge every N epochs (0=never)")
    parser.add_argument("--rerandomize-B", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=False)
    test_set = torchvision.datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)

    layer_sizes = [784] + args.hidden_sizes + [10]
    n_layers = len(layer_sizes) - 1
    ranks_per_layer = [args.ranks for _ in range(n_layers)] if args.ranks else None
    model = DecomposedMLP(layer_sizes, ranks_per_layer).to(device)

    print(f"Model: {layer_sizes}, ranks={args.ranks or 'none'}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        if args.merge_every > 0 and epoch > 1 and (epoch - 1) % args.merge_every == 0:
            model.merge_all(rerandomize_B=args.rerandomize_B)
            reset_factor_optimizer_state(optimizer, model)
            print(f"  [Merged at epoch {epoch}]")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train: {100*train_acc:.2f}% | "
            f"Test: {100*test_acc:.2f}%"
        )


if __name__ == "__main__":
    main()
