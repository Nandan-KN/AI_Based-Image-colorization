"""
train.py -- Training script for the LAB colorization U-Net.

Usage:
    # Smoke test (1 epoch, 10 images synthetic data):
    python train.py --smoke-test

    # Full training (downloads CIFAR-10 subset as training images):
    python train.py --epochs 5 --batch-size 16 --image-size 256
"""

import os
import sys
import argparse
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from dataset import ColorizationDataset, lab_tensors_to_rgb
from model import ColorizationUNet
from generate_dataset import generate_synthetic_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"  [CKPT] Saved checkpoint -> {path}")


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# Synthetic dataset (smoke test only -- no disk images needed)
# ---------------------------------------------------------------------------

class SyntheticColorizationDataset(Dataset):
    """
    Generates synthetic (L, ab) tensors on the fly.
    Used exclusively for --smoke-test to avoid downloading data.
    """

    def __init__(self, num_samples: int = 10, image_size: int = 256):
        self.num_samples = num_samples
        self.image_size  = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        H = W = self.image_size
        # L in [0, 1], ab in [-1, 1] -- same normalisation as ColorizationDataset
        L  = torch.rand(1, H, W)
        ab = torch.rand(2, H, W) * 2.0 - 1.0
        return L, ab


# ---------------------------------------------------------------------------
# Dataset preparation (synthetic images, no internet required)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (L, ab) in enumerate(loader):
        L  = L.to(device)
        ab = ab.to(device)

        optimizer.zero_grad()
        ab_pred = model(L)           # (B, 2, H, W)
        loss     = criterion(ab_pred, ab)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
            print(f"    Batch [{batch_idx + 1}/{len(loader)}] | Loss: {loss.item():.4f}", flush=True)

    avg_loss = running_loss / len(loader)
    print(f"  Epoch [{epoch}/{total_epochs}] | Loss: {avg_loss:.6f}", flush=True)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    for L, ab in loader:
        L  = L.to(device)
        ab = ab.to(device)
        ab_pred = model(L)
        running_loss += criterion(ab_pred, ab).item()
    return running_loss / len(loader)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LAB colorization U-Net")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Quick 1-epoch run on 10 synthetic samples")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--image-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max-images",  type=int,   default=500,
                        help="Max CIFAR-10 images to use for full training")
    parser.add_argument("--checkpoint",  type=str,   default="colorizer.pth")
    parser.add_argument("--data-dir",    type=str,
                        default=os.path.join(os.path.dirname(__file__), "data", "train"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")

    # ---- Dataset & DataLoader ----
    if args.smoke_test:
        print("[TRAIN] Running SMOKE TEST (synthetic data, 1 epoch, 10 samples)")
        train_dataset = SyntheticColorizationDataset(num_samples=10,
                                                      image_size=args.image_size)
        val_dataset   = SyntheticColorizationDataset(num_samples=4,
                                                      image_size=args.image_size)
        epochs = 1
    else:
        # Generate synthetic images (fast, no internet required)
        generate_synthetic_images(args.data_dir, num_images=args.max_images)
        full_dataset = ColorizationDataset(args.data_dir, image_size=args.image_size)
        if args.max_images and len(full_dataset) > args.max_images:
            full_dataset = torch.utils.data.Subset(full_dataset, range(args.max_images))
        val_split    = max(1, int(len(full_dataset) * 0.1))
        train_size   = len(full_dataset) - val_split
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_split],
            generator=torch.Generator().manual_seed(42),
        )
        epochs = args.epochs
        print(f"[TRAIN] Dataset: {len(full_dataset)} images "
              f"(train={train_size}, val={val_split})", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ---- Model, Loss, Optimizer ----
    model     = ColorizationUNet(base_channels=64).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TRAIN] Trainable parameters: {n_params:,}")
    print(f"[TRAIN] Epochs: {epochs} | Batch size: {args.batch_size} | LR: {args.lr}")

    # ---- Training ----
    best_val_loss  = math.inf
    loss_log       = []

    ckpt_path = os.path.join(os.path.dirname(__file__), args.checkpoint)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, epoch, epochs)
        val_loss   = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        loss_log.append((epoch, train_loss, val_loss))
        print(f"  Val Loss: {val_loss:.6f}  |  Elapsed: {elapsed:.1f}s", flush=True)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, ckpt_path)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("Training complete.")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Checkpoint    : {ckpt_path}")
    print("=" * 60, flush=True)

    if args.smoke_test:
        # Verify no NaN in the loss
        for ep, tl, vl in loss_log:
            if math.isnan(tl) or math.isnan(vl):
                print("[FAIL] Smoke test FAILED: NaN detected in loss!")
                sys.exit(1)
        print()
        print("[PASS] Smoke test passed!")

    # Print loss log
    print("\nEpoch | Train Loss | Val Loss")
    print("-" * 35)
    for ep, tl, vl in loss_log:
        print(f"  {ep:3d} |  {tl:.6f} | {vl:.6f}")


if __name__ == "__main__":
    main()
