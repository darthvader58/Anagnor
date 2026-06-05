"""Train the Anagnor landslide classifier.

Logs per-epoch train/val loss and accuracy to `checkpoints/metrics.json`, which
visualize.py reads to plot the training-vs-validation curves.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from datasetGenerator import CustomDataset
from model import AnagnorModel


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def run_epoch(net, loader, criterion, optimizer, device, train: bool):
    net.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += _accuracy(logits, labels)
            n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = CustomDataset()
    n_val = int(len(full) * args.val_split)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net = AnagnorModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(net, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(net, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f}  |  "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        with open(os.path.join(args.checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "best.pth"))

    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "final.pth"))
    print(f"Finished training. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
