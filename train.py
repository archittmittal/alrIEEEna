"""
train.py — Core training loop for the IEEE GEHU hackathon ML pipeline.

Features:
  - Mixed Precision (AMP) training
  - Cosine LR schedule with linear warmup
  - AdamW optimizer
  - Label-smoothed cross-entropy + class weights
  - Mixup / CutMix alternating augmentation
  - WeightedRandomSampler (from dataset.py)
  - Early stopping
  - Best checkpoint saving per backbone

Usage:
    python train.py                     # trains all backbones in config.BACKBONES
    python train.py --backbone efficientnetv2_l.in21k   # train one backbone
    python train.py --pseudo            # also load pseudo-labels from pseudo_labels.csv
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import config
from dataset import get_train_val_loaders, load_csv
from model import build_model, LabelSmoothingCrossEntropy
from utils import (
    AverageMeter, accuracy, compute_class_weights,
    mixup_data, mixup_criterion, cutmix_data, set_seed, save_checkpoint,
)


# ─────────────────────────────────────────────────────────────
# Warmup + Cosine LR scheduler
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int, steps_per_epoch: int):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to MIN_LR.
    Using PyTorch's LambdaLR for step-level granularity.
    """
    total_steps  = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        # Scale between MIN_LR/LR and 1.0
        min_ratio = config.MIN_LR / config.LR
        return max(min_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, epoch):
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc@1")
    use_mixup  = config.MIXUP_ALPHA > 0
    use_cutmix = config.CUTMIX_ALPHA > 0

    for step, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Decide: apply Mixup or CutMix randomly (50/50 if both enabled)
        apply_mix = False
        if use_mixup or use_cutmix:
            r = random.random()
            if use_mixup and use_cutmix:
                if r < 0.5:
                    images, y_a, y_b, lam = mixup_data(images, targets, config.MIXUP_ALPHA)
                else:
                    images, y_a, y_b, lam = cutmix_data(images, targets, config.CUTMIX_ALPHA)
                apply_mix = True
            elif use_mixup:
                images, y_a, y_b, lam = mixup_data(images, targets, config.MIXUP_ALPHA)
                apply_mix = True
            elif use_cutmix:
                images, y_a, y_b, lam = cutmix_data(images, targets, config.CUTMIX_ALPHA)
                apply_mix = True

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.USE_AMP):
            logits = model(images)
            if apply_mix:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accuracy (use original targets for logging even with mixup)
        with torch.no_grad():
            top1, = accuracy(logits, targets, topk=(1,))
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))

        if step % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [Epoch {epoch:02d}] Step {step:04d}/{len(loader):04d} | "
                  f"loss={loss_meter.avg:.4f} | acc={acc_meter.avg:.2f}% | lr={lr_now:.2e}")

    return loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter  = AverageMeter("val_acc@1")
    acc5_meter = AverageMeter("val_acc@5")

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=config.USE_AMP):
            logits = model(images)
            loss   = criterion(logits, targets)

        top1, top5 = accuracy(logits, targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))
        acc5_meter.update(top5, images.size(0))

    return loss_meter.avg, acc_meter.avg, acc5_meter.avg


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train_backbone(backbone_name: str, pseudo_csv: str = None):
    """
    Train a single backbone and save best checkpoint.

    Args:
        backbone_name : timm model name
        pseudo_csv    : path to pseudo-label CSV (IMAGE, LABEL) or None
    """
    set_seed(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training: {backbone_name}")
    print(f"  Device:   {device}")
    print(f"{'='*60}\n")

    img_size = config.IMG_SIZE_MAP.get(backbone_name, config.DEFAULT_IMG_SIZE)

    # Load pseudo-labels if provided
    extra_images, extra_labels = None, None
    if pseudo_csv and os.path.exists(pseudo_csv):
        extra_images, extra_labels = load_csv(pseudo_csv)
        print(f"[Pseudo] Loaded {len(extra_images)} pseudo-labelled samples from {pseudo_csv}")

    train_loader, val_loader, train_labels = get_train_val_loaders(
        img_size, extra_images, extra_labels
    )

    # Model
    model = build_model(backbone_name, pretrained=True).to(device)

    # Loss — weighted + label smoothing
    class_weights = compute_class_weights(train_labels, config.NUM_CLASSES, device)
    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.LABEL_SMOOTHING,
        weight=class_weights,
    )

    # Optimizer — separate LR for backbone vs head for fine-tuning stability
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config.LR * 0.1},  # lower LR for pretrained weights
        {"params": head_params,     "lr": config.LR},
    ], weight_decay=config.WEIGHT_DECAY)

    scaler    = GradScaler(enabled=config.USE_AMP)
    scheduler = build_scheduler(
        optimizer,
        num_epochs=config.EPOCHS,
        warmup_epochs=config.WARMUP_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # Checkpoint path
    safe_name = backbone_name.replace("/", "_").replace(".", "_")
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"best_{safe_name}.pth")

    best_val_acc   = 0.0
    no_improve_cnt = 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{config.EPOCHS} ──────────────────────────────")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, device, epoch
        )
        val_loss, val_acc, val_acc5 = validate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_acc5": val_acc5,
        })

        print(f"\n  Summary Epoch {epoch:02d}: "
              f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} | val_acc@1={val_acc:.2f}% | val_acc@5={val_acc5:.2f}%")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc   = val_acc
            no_improve_cnt = 0
            save_checkpoint({
                "epoch":           epoch,
                "backbone":        backbone_name,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc":        best_val_acc,
            }, ckpt_path)
            print(f"  ✅ New best val acc: {best_val_acc:.2f}% — checkpoint saved.")
        else:
            no_improve_cnt += 1
            print(f"  No improvement ({no_improve_cnt}/{config.EARLY_STOP_PATIENCE})")

        if no_improve_cnt >= config.EARLY_STOP_PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch}. Best val acc: {best_val_acc:.2f}%")
            break

    # Save training history as CSV
    import csv
    hist_path = os.path.join(config.OUTPUT_DIR, f"history_{safe_name}.csv")
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\n[Train] History saved → {hist_path}")
    print(f"[Train] Best checkpoint → {ckpt_path}")
    print(f"[Train] Best Val Acc: {best_val_acc:.2f}%\n")
    return best_val_acc


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IEEE GEHU hackathon model")
    parser.add_argument("--backbone", type=str, default=None,
                        help="timm backbone name. If omitted, trains all in config.BACKBONES")
    parser.add_argument("--pseudo", action="store_true",
                        help="Include pseudo-labels from outputs/pseudo_labels.csv")
    args = parser.parse_args()

    pseudo_csv = os.path.join(config.OUTPUT_DIR, "pseudo_labels.csv") if args.pseudo else None

    backbones = [args.backbone] if args.backbone else config.BACKBONES
    results   = {}
    for bb in backbones:
        acc = train_backbone(bb, pseudo_csv=pseudo_csv)
        results[bb] = acc

    print("\n" + "="*60)
    print("  TRAINING COMPLETE — Summary")
    print("="*60)
    for bb, acc in results.items():
        print(f"  {bb:55s}  →  {acc:.2f}%")
    print("="*60)
