"""
train.py — Training loop for GCPNet.

Usage (Colab / Kaggle):
  python train.py \
    --data_root /path/to/train_dataset \
    --json      /path/to/train_dataset/curated_gcp_marks.json \
    --epochs    40 \
    --batch_size 16 \
    --backbone  efficientnet_b2 \
    --output_dir ./runs/exp1

The script:
  1. Builds dataloaders with stratified val split
  2. Trains with cosine LR + warmup
  3. Logs train/val metrics each epoch
  4. Saves best checkpoint by val PCK@25px
  5. Saves final predictions dict for easy resumption
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score
import numpy as np

from dataset import build_dataloaders, SHAPE_CLASSES, CROP_SIZE
from model import GCPNet, GCPLoss, compute_pck


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_device(batch, device):
    imgs, coords, labels, meta = batch
    return imgs.to(device), coords.to(device), labels.to(device), meta


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = reg_loss_sum = cls_loss_sum = 0.0
    n = 0

    for batch in loader:
        imgs, gt_coords, gt_labels, _ = to_device(batch, device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred_coords, pred_logits = model(imgs)
            loss, l_reg, l_cls = criterion(pred_coords, pred_logits,
                                           gt_coords, gt_labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()

        bs = imgs.size(0)
        total_loss    += loss.item()  * bs
        reg_loss_sum  += l_reg.item() * bs
        cls_loss_sum  += l_cls.item() * bs
        n += bs

    return {
        "loss":     total_loss   / n,
        "reg_loss": reg_loss_sum / n,
        "cls_loss": cls_loss_sum / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred_abs, all_gt_abs = [], []
    all_pred_cls, all_gt_cls = [], []
    n = 0

    for batch in loader:
        imgs, gt_coords, gt_labels, meta = to_device(batch, device)

        with torch.cuda.amp.autocast():
            pred_coords, pred_logits = model(imgs)
            loss, _, _ = criterion(pred_coords, pred_logits, gt_coords, gt_labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

        # De-normalise for PCK: we need crop offsets.
        # During validation crops are centred on GCP so crop_x1 = abs_cx - 256, etc.
        abs_cx = torch.tensor([float(v) for v in meta["abs_cx"]], device=device)
        abs_cy = torch.tensor([float(v) for v in meta["abs_cy"]], device=device)
        crop_x1 = abs_cx - CROP_SIZE // 2
        crop_y1 = abs_cy - CROP_SIZE // 2

        # pred_coords in [-1,1] → pixel in crop → absolute
        px = (pred_coords[:, 0] + 1.0) / 2.0 * CROP_SIZE + crop_x1
        py = (pred_coords[:, 1] + 1.0) / 2.0 * CROP_SIZE + crop_y1
        pred_abs = torch.stack([px, py], dim=1)
        gt_abs   = torch.stack([abs_cx, abs_cy], dim=1)

        all_pred_abs.append(pred_abs.cpu())
        all_gt_abs.append(gt_abs.cpu())
        all_pred_cls.append(pred_logits.argmax(dim=1).cpu())
        all_gt_cls.append(gt_labels.cpu())

    all_pred_abs = torch.cat(all_pred_abs)
    all_gt_abs   = torch.cat(all_gt_abs)
    all_pred_cls = torch.cat(all_pred_cls).numpy()
    all_gt_cls   = torch.cat(all_gt_cls).numpy()

    pck_metrics = compute_pck(all_pred_abs, all_gt_abs)
    f1 = f1_score(all_gt_cls, all_pred_cls, average="macro", zero_division=0)
    cls_acc = (all_pred_cls == all_gt_cls).mean()

    return {
        "loss":     total_loss / n,
        "f1_macro": f1,
        "cls_acc":  cls_acc,
        **pck_metrics,
    }


def print_metrics(epoch, train_m, val_m, lr, elapsed):
    print(
        f"Epoch {epoch:3d} | "
        f"Train loss {train_m['loss']:.4f} (reg {train_m['reg_loss']:.4f} cls {train_m['cls_loss']:.4f}) | "
        f"Val loss {val_m['loss']:.4f} | "
        f"PCK@10={val_m['PCK@10px']:.3f} @25={val_m['PCK@25px']:.3f} @50={val_m['PCK@50px']:.3f} | "
        f"dist={val_m['mean_dist_px']:.1f}px | "
        f"F1={val_m['f1_macro']:.3f} acc={val_m['cls_acc']:.3f} | "
        f"LR={lr:.2e} | {elapsed:.0f}s"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--json",        required=True)
    parser.add_argument("--output_dir",  default="./runs/exp1")
    parser.add_argument("--backbone",    default="efficientnet_b2")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--val_frac",    type=float, default=0.15)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--lambda_reg",  type=float, default=1.0)
    parser.add_argument("--lambda_cls",  type=float, default=1.0)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--fp16",        action="store_true", default=True)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, stats = build_dataloaders(
        args.data_root, args.json,
        val_fraction=args.val_frac,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GCPNet(
        backbone_name=args.backbone,
        num_classes=len(SHAPE_CLASSES),
        pretrained=True,
        dropout=args.dropout,
    ).to(device)

    class_weights = stats["class_weights"].to(device)
    criterion = GCPLoss(class_weights, args.lambda_reg, args.lambda_cls)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler() if (args.fp16 and device.type == "cuda") else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_pck25 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion,
                                  optimizer, scheduler, device, scaler)
        val_m   = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        print_metrics(epoch, train_m, val_m, current_lr, elapsed)

        history.append({"epoch": epoch, **train_m, **{f"val_{k}": v for k, v in val_m.items()}})

        # Save best
        if val_m["PCK@25px"] > best_pck25:
            best_pck25 = val_m["PCK@25px"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_metrics": val_m,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best model (PCK@25={best_pck25:.3f}) → {ckpt_path}")

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "final_model.pth"))

    # Save history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best PCK@25px = {best_pck25:.4f}")
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
