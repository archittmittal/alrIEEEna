"""
utils.py — Shared utilities for the IEEE GEHU hackathon ML pipeline.
"""

import os
import random
import numpy as np
import torch
from collections import Counter


# ─────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# 2. Class weights for imbalanced dataset
# ─────────────────────────────────────────────
def compute_class_weights(labels: list, num_classes: int, device: str = "cpu") -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    Minority classes get higher weight → loss focuses more on them.
    """
    counts = Counter(labels)
    total  = len(labels)
    weights = []
    for cls in range(num_classes):
        cnt = counts.get(cls, 1)          # avoid division by zero
        weights.append(total / (num_classes * cnt))
    weights = torch.tensor(weights, dtype=torch.float32)
    # Clip extreme weights so one class doesn't dominate
    weights = weights.clamp(min=0.1, max=10.0)
    return weights.to(device)


# ─────────────────────────────────────────────
# 3. Running average meter
# ─────────────────────────────────────────────
class AverageMeter:
    """Computes and stores the running average of a metric."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


# ─────────────────────────────────────────────
# 4. Accuracy computation
# ─────────────────────────────────────────────
def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    """
    Compute top-k accuracy. Returns list of scalars (one per k).
    Works correctly even if num_classes < max(topk).
    """
    with torch.no_grad():
        maxk = min(max(topk), outputs.size(1))
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k_use = min(k, maxk)
            correct_k = correct[:k_use].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# ─────────────────────────────────────────────
# 5. Checkpoint helpers
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, filepath: str):
    """Save model checkpoint to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"[Checkpoint] Saved → {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None):
    """Load model (and optionally optimizer) state from checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_acc", 0.0)
    print(f"[Checkpoint] Loaded from {filepath}  (epoch {epoch}, best_acc {best_acc:.2f}%)")
    return epoch, best_acc


# ─────────────────────────────────────────────
# 6. Mixup / CutMix helpers
# ─────────────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    Mixed loss = λ * CE(y_a) + (1-λ) * CE(y_b)
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    CutMix: paste a random box from one image into another.
    Returns mixed images, targets, and effective lambda.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    H, W = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # Adjust effective lambda based on actual box area
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed_x, y, y[index], lam
