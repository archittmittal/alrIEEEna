"""
══════════════════════════════════════════════════════════════════
IEEE GEHU — alrIEEEna26 ML Challenge · HACKATHON NOTEBOOK
GPU must be ON: Settings → Accelerator → GPU T4 x2
ALL APIs verified for: timm 1.0+ | albumentations 2.x | torch 2.x
══════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────
# ── CELL 1 ── Paths, CSV loading & quick sanity check
# (Keep exactly as given — do NOT change)
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# NOTE: Add Input Dataset [alrieeena26-ml-challenge-by-ieee-sb-gehu] Before Use.

# Path Definition
base_path      = '/kaggle/input/datasets/ieeesbgehu/alrieeena26-ml-challenge-by-ieee-sb-gehu/ML FINAL DATASET/'
train_csv_path = os.path.join(base_path, 'TRAIN.csv')
test_csv_path  = os.path.join(base_path, 'TEST.csv')
image_dir      = os.path.join(base_path, 'images/')

# Verify CSVs Successfully Imported
try:
    test_df  = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(train_csv_path)
    print("✅ TRAIN.CSV and TEST.CSV loaded successfully!")
    print(f"   TRAIN rows: {len(train_df):,}  |  TEST rows: {len(test_df):,}")
except Exception as e:
    print("❌ Error loading CSVs:", e)

# Sample Images to Verify Images Successfully Imported
fig, axes = plt.subplots(1, 2, figsize=(4, 2))

train_img_path   = os.path.join(image_dir, train_df['IMAGE'].iloc[0])
sample_train_img = plt.imread(train_img_path)
axes[0].imshow(sample_train_img)
axes[0].set_title(f"TRAIN: {train_df['IMAGE'].iloc[0]}")
axes[0].axis("off")

test_img_path   = os.path.join(image_dir, test_df['IMAGE'].iloc[0])
sample_test_img = plt.imread(test_img_path)
axes[1].imshow(sample_test_img)
axes[1].set_title(f"TEST: {test_df['IMAGE'].iloc[0]}")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# ── CELL 2 ── EDA & Visualizations
# ─────────────────────────────────────────────────────────────

from collections import Counter
import random

print("=" * 55)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 55)

labels      = train_df['LABEL'].astype(int).tolist()
label_cnt   = Counter(labels)
num_classes = len(label_cnt)

print(f"\n📊 Dataset Overview")
print(f"  Training samples : {len(train_df):,}")
print(f"  Test   samples   : {len(test_df):,}")
print(f"  Classes          : {num_classes}")
print(f"  Label range      : {min(labels)} → {max(labels)}")
print(f"  Avg imgs/class   : {len(labels)/num_classes:.1f}")
print(f"  Max imgs/class   : {max(label_cnt.values())} (class {max(label_cnt, key=label_cnt.get)})")
print(f"  Min imgs/class   : {min(label_cnt.values())} (class {min(label_cnt, key=label_cnt.get)})")
print(f"  Imbalance ratio  : {max(label_cnt.values())/min(label_cnt.values()):.1f}×")

# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
counts_sorted = sorted(label_cnt.values(), reverse=True)
axes[0].bar(range(len(counts_sorted)), counts_sorted, color='steelblue', width=1)
axes[0].set_title("Class Distribution (sorted by frequency)", fontsize=12)
axes[0].set_xlabel("Class rank"); axes[0].set_ylabel("# Images")
axes[0].axhline(np.mean(counts_sorted), color='red', linestyle='--',
                label=f"Mean={np.mean(counts_sorted):.0f}"); axes[0].legend()
axes[1].hist(counts_sorted, bins=30, color='coral', edgecolor='white')
axes[1].set_title("Samples per class histogram", fontsize=12)
axes[1].set_xlabel("# Images per class"); axes[1].set_ylabel("# Classes")
plt.suptitle("Class Imbalance Analysis", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Sample images grid
sample_classes = sorted(label_cnt.keys())[:16]
fig, axes = plt.subplots(2, 8, figsize=(20, 5))
for ax, cls in zip(axes.flat, sample_classes):
    row  = train_df[train_df['LABEL'] == cls].iloc[0]
    path = os.path.join(image_dir, row['IMAGE'])
    try:
        img = PILImage.open(path).convert("RGB").resize((128, 128))
        ax.imshow(np.array(img))
    except Exception:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
    ax.set_title(f"Cls {cls}", fontsize=8); ax.axis("off")
plt.suptitle("Sample Images (first 16 classes)", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Image size distribution
random.seed(42)
sample_imgs = random.sample(train_df['IMAGE'].tolist(), min(300, len(train_df)))
widths, heights = [], []
for fn in sample_imgs:
    try:
        w, h = PILImage.open(os.path.join(image_dir, fn)).size
        widths.append(w); heights.append(h)
    except Exception:
        pass
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].hist(widths, bins=30, color='mediumseagreen', edgecolor='white')
axes[0].set_title("Image Width Distribution"); axes[0].set_xlabel("px")
axes[1].hist(heights, bins=30, color='mediumpurple', edgecolor='white')
axes[1].set_title("Image Height Distribution"); axes[1].set_xlabel("px")
plt.suptitle(f"Image sizes (n={len(widths)})", fontweight='bold')
plt.tight_layout(); plt.show()
print(f"  Width  → mean={np.mean(widths):.0f}  min={min(widths)}  max={max(widths)}")
print(f"  Height → mean={np.mean(heights):.0f}  min={min(heights)}  max={max(heights)}")
print("\n✅ EDA done")


# ─────────────────────────────────────────────────────────────
# ── CELL 3 ── Install packages & verify versions
# ─────────────────────────────────────────────────────────────

import subprocess, sys

# Install/upgrade packages
for pkg in ["timm", "albumentations", "opencv-python-headless"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg], check=True)

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import cv2
print(f"timm         : {timm.__version__}")
print(f"torch        : {torch.__version__}")
print(f"albumentations: {A.__version__}")
print(f"cv2          : {cv2.__version__}")

# ── Verify backbone name at import time ───────────────────────
# List all EfficientNetV2-L variants available in this timm version
available = timm.list_models("*efficientnetv2_l*", pretrained=True)
print(f"\nAvailable EfficientNetV2-L pretrained models:")
for m in available: print(f"  {m}")

# Pick best available backbone (priority order)
PREFERRED_BACKBONES = [
    "tf_efficientnetv2_l.in21k_ft_in1k",   # best accuracy, IN21K→IN1K
    "tf_efficientnetv2_l.in1k",             # good fallback
    "efficientnetv2_l",                      # base without tag
    "tf_efficientnetv2_l",                   # tf variant no tag
]
CHOSEN_BACKBONE = None
for b in PREFERRED_BACKBONES:
    if b in available or b in timm.list_models(b):
        CHOSEN_BACKBONE = b
        break
if CHOSEN_BACKBONE is None:
    # Ultimate fallback — always available
    CHOSEN_BACKBONE = "tf_efficientnetv2_l.in21k_ft_in1k"

print(f"\n✅ Using backbone: {CHOSEN_BACKBONE}")

# ── AMP imports (compatible with PyTorch 2.x) ─────────────────
from torch.amp import autocast, GradScaler

# ── Other imports ─────────────────────────────────────────────
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Global config ─────────────────────────────────────────────
CFG = dict(
    backbone         = CHOSEN_BACKBONE,
    img_size         = 384,
    batch_size       = 12,        # safe for T4 16GB at 384px
    num_workers      = 2,
    num_classes      = 397,
    val_split        = 0.15,
    seed             = 42,
    epochs           = 25,
    lr               = 3e-4,
    min_lr           = 1e-6,
    weight_decay     = 1e-2,
    grad_clip        = 5.0,
    warmup_epochs    = 2,
    label_smoothing  = 0.1,
    mixup_alpha      = 0.4,
    cutmix_alpha     = 0.4,
    early_stop       = 8,
    tta_steps        = 8,
    pseudo_threshold = 0.92,
    pseudo_epochs    = 8,
)
OUTPUT  = Path("/kaggle/working")
IMG_DIR = Path(image_dir)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CFG["seed"])
print(f"\n✅ Config ready  |  backbone={CFG['backbone']}  img_size={CFG['img_size']}")


# ─────────────────────────────────────────────────────────────
# ── CELL 4 ── Augmentation pipelines (albumentations v2.x)
# ─────────────────────────────────────────────────────────────

def get_train_transforms(size: int):
    h = max(4, size // 8)      # hole size for CoarseDropout
    return A.Compose([
        # size must be tuple (H, W) in albumentations v2
        A.RandomResizedCrop(size=(size, size), scale=(0.65, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.2),
        # CoarseDropout v2 API: range-based params
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(h // 2, h),
            hole_width_range=(h // 2, h),
            fill_value=0, p=0.3
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(size: int):
    big = int(size * 1.1)
    return A.Compose([
        A.Resize(height=big, width=big),       # keyword args required in v2
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# TTA: 8 different augmentation views
TTA_AUGMENTS = [
    lambda s: [],                                               # 0: clean
    lambda s: [A.HorizontalFlip(p=1.0)],                       # 1: h-flip
    lambda s: [A.VerticalFlip(p=1.0)],                         # 2: v-flip
    lambda s: [A.Transpose(p=1.0)],                            # 3: transpose
    lambda s: [A.Rotate(limit=15, p=1.0,
                        border_mode=cv2.BORDER_REFLECT)],       # 4: rotate±15
    lambda s: [A.RandomCrop(height=s, width=s)],               # 5: random crop
    lambda s: [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],# 6: both flips
    lambda s: [A.Rotate(limit=30, p=1.0,
                        border_mode=cv2.BORDER_REFLECT)],       # 7: rotate±30
]


def get_tta_transforms(size: int, idx: int):
    big   = int(size * 1.1)
    base  = [A.Resize(height=big, width=big), A.CenterCrop(height=size, width=size)]
    extra = TTA_AUGMENTS[idx % len(TTA_AUGMENTS)](size)
    return A.Compose(
        base + extra + [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ── Quick smoke-test the transforms ──────────────────────────
_dummy = np.zeros((500, 400, 3), dtype=np.uint8)
_ = get_train_transforms(CFG["img_size"])(image=_dummy)
_ = get_val_transforms(CFG["img_size"])(image=_dummy)
_ = get_tta_transforms(CFG["img_size"], 0)(image=_dummy)
print("✅ All augmentation transforms verified — no API errors")


# ─────────────────────────────────────────────────────────────
# ── CELL 5 ── Dataset & DataLoaders
# ─────────────────────────────────────────────────────────────

class HackDataset(Dataset):
    def __init__(self, img_names, labels, img_dir, transform):
        self.img_names = img_names
        self.labels    = labels          # None for test set
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        path = str(self.img_dir / self.img_names[idx])
        img  = cv2.imread(path)
        if img is None:
            # cv2 failed (rare) — fall back to PIL
            img = np.array(PILImage.open(path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        if self.labels is not None:
            return img, torch.tensor(self.labels[idx], dtype=torch.long)
        return img, self.img_names[idx]


def build_loaders(extra_images=None, extra_labels=None):
    images = train_df['IMAGE'].tolist()
    lbls   = train_df['LABEL'].astype(int).tolist()

    if extra_images:
        images = images + extra_images
        lbls   = lbls + extra_labels
        print(f"  + {len(extra_images):,} pseudo-labelled samples")

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=CFG["val_split"], random_state=CFG["seed"]
    )
    tr_idx, va_idx = next(sss.split(images, lbls))
    tr_imgs = [images[i] for i in tr_idx]; tr_lbls = [lbls[i] for i in tr_idx]
    va_imgs = [images[i] for i in va_idx]; va_lbls = [lbls[i] for i in va_idx]
    print(f"  Train: {len(tr_imgs):,}  |  Val: {len(va_imgs):,}")

    # Class-weighted sampler (fights imbalance)
    cc = [0] * CFG["num_classes"]
    for l in tr_lbls: cc[l] += 1
    sw = [1.0 / max(cc[l], 1) for l in tr_lbls]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(tr_lbls), replacement=True)

    sz    = CFG["img_size"]
    tr_ds = HackDataset(tr_imgs, tr_lbls, IMG_DIR, get_train_transforms(sz))
    va_ds = HackDataset(va_imgs, va_lbls, IMG_DIR, get_val_transforms(sz))

    tr_dl = DataLoader(tr_ds, batch_size=CFG["batch_size"], sampler=sampler,
                       num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                       num_workers=CFG["num_workers"], pin_memory=True)
    return tr_dl, va_dl, tr_lbls


def get_test_loader(tta_idx: int = 0):
    imgs = test_df['IMAGE'].tolist()
    ds   = HackDataset(imgs, None, IMG_DIR, get_tta_transforms(CFG["img_size"], tta_idx))
    dl   = DataLoader(ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                      num_workers=CFG["num_workers"], pin_memory=True)
    return dl, imgs

print("✅ Dataset classes ready")


# ─────────────────────────────────────────────────────────────
# ── CELL 6 ── Model + Loss + Utilities
# ─────────────────────────────────────────────────────────────

class HackModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 397, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        # Proper weight init
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class LabelSmoothCE(nn.Module):
    """Cross-entropy with label smoothing + optional class weights."""
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n  = logits.size(-1)
        lp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            y = torch.full_like(lp, self.smoothing / (n - 1))
            y.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = (-y * lp).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def compute_class_weights(labels) -> torch.Tensor:
    cnt = Counter(labels); tot = len(labels); n = CFG["num_classes"]
    w   = [tot / (n * max(cnt.get(c, 1), 1)) for c in range(n)]
    return torch.tensor(w, dtype=torch.float32).clamp(0.1, 10.0).to(DEVICE)


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(CFG["min_lr"] / CFG["lr"], 0.5 * (1 + np.cos(np.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3)
    cr = np.sqrt(1 - lam)
    ch, cw = int(H * cr), int(W * cr)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cw // 2, 0, W); x2 = np.clip(cx + cw // 2, 0, W)
    y1 = np.clip(cy - ch // 2, 0, H); y2 = np.clip(cy + ch // 2, 0, H)
    mx = x.clone(); mx[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    return mx, y, y[idx], 1 - (x2 - x1) * (y2 - y1) / (W * H)

def mix_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)

def top1_acc(logits, targets):
    return logits.argmax(1).eq(targets).float().mean().item() * 100

# ── Smoke-test model creation ─────────────────────────────────
print(f"Loading backbone: {CFG['backbone']} (pretrained=False for test)...")
_m = HackModel(CFG["backbone"], CFG["num_classes"], pretrained=False)
_x = torch.randn(2, 3, 64, 64)
_o = _m(_x)
assert _o.shape == (2, CFG["num_classes"]), f"Bad output shape: {_o.shape}"
del _m, _x, _o
print(f"✅ Model OK — output shape verified ({CFG['num_classes']} classes)")
print("✅ All utilities ready")


# ─────────────────────────────────────────────────────────────
# ── CELL 7 ── Training loop  (~60-70 min on T4)
# ─────────────────────────────────────────────────────────────

def run_training(extra_images=None, extra_labels=None, warm_start=None):
    set_seed(CFG["seed"])
    print(f"\n{'='*60}")
    print(f"  Backbone : {CFG['backbone']}")
    print(f"  Img size : {CFG['img_size']}  |  Batch: {CFG['batch_size']}")
    print(f"  Epochs   : {CFG['epochs']}    |  Device: {DEVICE}")
    print(f"{'='*60}\n")

    tr_dl, va_dl, tr_lbls = build_loaders(extra_images, extra_labels)

    # Build model — pretrained only on first run, warm-start otherwise
    model = HackModel(
        CFG["backbone"], CFG["num_classes"],
        pretrained=(warm_start is None)
    ).to(DEVICE)

    if warm_start is not None:
        ck_path = OUTPUT / warm_start
        if ck_path.exists():
            ck = torch.load(str(ck_path), map_location="cpu")
            model.load_state_dict(ck["model_state"])
            print(f"  ↳ Warm-started from {warm_start}  (prev best={ck.get('best_acc',0):.2f}%)")

    cw        = compute_class_weights(tr_lbls)
    criterion = LabelSmoothCE(CFG["label_smoothing"], weight=cw)

    # Lower LR for pretrained backbone, full LR for our custom head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": CFG["lr"] * 0.1},
        {"params": model.head.parameters(),     "lr": CFG["lr"]},
    ], weight_decay=CFG["weight_decay"])

    total_steps  = CFG["epochs"] * len(tr_dl)
    warmup_steps = CFG["warmup_epochs"] * len(tr_dl)
    scheduler    = build_lr_scheduler(optimizer, total_steps, warmup_steps)

    # torch.amp.GradScaler (PyTorch 2.x API)
    scaler = GradScaler("cuda") if DEVICE.type == "cuda" else GradScaler("cpu")

    best_acc, no_imp = 0.0, 0
    ckpt_path = OUTPUT / "best_model.pth"
    hist = []

    for epoch in range(1, CFG["epochs"] + 1):

        # ─── Train ───────────────────────────────────────────
        model.train()
        tl, ta, ns = 0.0, 0.0, 0
        pbar = tqdm(tr_dl, desc=f"Ep {epoch:02d}/{CFG['epochs']} [Train]", leave=False)
        for imgs, targets in pbar:
            imgs    = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Alternate mixup and cutmix randomly
            if random.random() < 0.5:
                imgs, ya, yb, lam = mixup(imgs, targets, CFG["mixup_alpha"])
            else:
                imgs, ya, yb, lam = cutmix(imgs, targets, CFG["cutmix_alpha"])

            optimizer.zero_grad(set_to_none=True)

            # torch.amp.autocast (PyTorch 2.x API)
            with autocast(device_type=DEVICE.type):
                out  = model(imgs)
                loss = mix_loss(criterion, out, ya, yb, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_acc = top1_acc(out, targets)
            tl += loss.item(); ta += batch_acc; ns += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{batch_acc:.1f}%",
                lr=f"{optimizer.param_groups[1]['lr']:.1e}"
            )

        # ─── Validate ────────────────────────────────────────
        model.eval()
        vl, va, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, targets in tqdm(va_dl, desc=f"Ep {epoch:02d}/{CFG['epochs']} [Val]  ", leave=False):
                imgs    = imgs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                with autocast(device_type=DEVICE.type):
                    out  = model(imgs)
                    loss = criterion(out, targets)
                vl += loss.item(); va += top1_acc(out, targets); vn += 1

        tr_l, tr_a = tl / ns, ta / ns
        va_l, va_a = vl / vn, va / vn
        hist.append(dict(epoch=epoch, tr_loss=tr_l, tr_acc=tr_a,
                         va_loss=va_l, va_acc=va_a))

        flag = ""
        if va_a > best_acc:
            best_acc = va_a; no_imp = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_acc": best_acc,
                "backbone": CFG["backbone"],
            }, str(ckpt_path))
            flag = "  ✅ SAVED"
        else:
            no_imp += 1
            flag = f"  (no improve {no_imp}/{CFG['early_stop']})"

        print(f"Ep {epoch:02d}  "
              f"tr_loss={tr_l:.4f} tr_acc={tr_a:.2f}%  "
              f"va_loss={va_l:.4f} va_acc={va_a:.2f}%{flag}")

        if no_imp >= CFG["early_stop"]:
            print(f"\n⏹  Early stop at epoch {epoch}. Best val acc: {best_acc:.2f}%")
            break

    # Plot training curves
    df_h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(df_h.epoch, df_h.tr_loss, label="Train"); ax[0].plot(df_h.epoch, df_h.va_loss, label="Val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].set_xlabel("Epoch")
    ax[1].plot(df_h.epoch, df_h.tr_acc,  label="Train"); ax[1].plot(df_h.epoch, df_h.va_acc,  label="Val")
    ax[1].set_title("Accuracy (%)"); ax[1].legend(); ax[1].set_xlabel("Epoch")
    plt.suptitle(f"Training — Best val acc: {best_acc:.2f}%", fontweight="bold")
    plt.tight_layout(); plt.show()

    print(f"\n🏁 Training done. Best val acc: {best_acc:.2f}%")
    return hist, best_acc

# ── START TRAINING ───────────────────────────────────────────
history, best_val_acc = run_training()


# ─────────────────────────────────────────────────────────────
# ── CELL 8 ── TTA Inference → FINAL.csv  ⚠️ DOWNLOAD AS BACKUP
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_tta(ckpt_name: str = "best_model.pth", tta_steps: int = None, save_probs: bool = True):
    tta_steps = tta_steps or CFG["tta_steps"]
    ck_path   = OUTPUT / ckpt_name
    ck        = torch.load(str(ck_path), map_location=DEVICE)

    model = HackModel(CFG["backbone"], CFG["num_classes"], pretrained=False).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"[Inference] Loaded {ckpt_name}  best_acc={ck.get('best_acc', 0):.2f}%  TTA×{tta_steps}")

    accum     = None
    all_files = None

    for t in range(tta_steps):
        dl, files = get_test_loader(tta_idx=t)
        if all_files is None:
            all_files = files
            accum     = np.zeros((len(files), CFG["num_classes"]), dtype=np.float64)

        batch_probs = []
        for imgs, _ in tqdm(dl, desc=f"  TTA {t+1}/{tta_steps}", leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type):
                out = model(imgs)
            batch_probs.append(torch.softmax(out, dim=1).cpu().numpy())
        accum += np.concatenate(batch_probs, axis=0)

    probs = accum / tta_steps
    preds = probs.argmax(axis=1).astype(int)
    confs = probs.max(axis=1)

    # Save FINAL.csv
    df_out = pd.DataFrame({"IMAGE": all_files, "LABEL": preds})
    df_out.to_csv(str(OUTPUT / "FINAL.csv"), index=False)

    # Sanity checks
    assert len(df_out) == len(test_df),                    "Row count mismatch!"
    assert list(df_out.columns) == ["IMAGE", "LABEL"],     "Column names wrong!"
    assert df_out["LABEL"].between(0, 396).all(),          "Label out of range!"
    assert (df_out["IMAGE"] == test_df["IMAGE"]).all(),    "Image order mismatch!"

    print(f"\n✅ FINAL.csv saved → /kaggle/working/FINAL.csv")
    print(f"   Rows           : {len(df_out):,}")
    print(f"   Unique classes : {df_out['LABEL'].nunique()} / {CFG['num_classes']}")
    print(f"   Mean confidence: {confs.mean():.4f}")
    print(f"   % conf > 0.90  : {(confs > 0.90).mean() * 100:.1f}%")
    print(f"\n⚠️  DOWNLOAD FINAL.csv NOW as backup before pseudo-labeling!")
    print("\nSample predictions:")
    print(df_out.head(5).to_string(index=False))

    if save_probs:
        np.save(str(OUTPUT / "test_probs.npy"),     probs)
        np.save(str(OUTPUT / "test_filenames.npy"), np.array(all_files))
        print("\n   Probabilities saved for pseudo-labeling.")

    return probs, all_files

probs, test_files = infer_tta()


# ─────────────────────────────────────────────────────────────
# ── CELL 9 ── Pseudo-label generation
# ─────────────────────────────────────────────────────────────

def gen_pseudo_labels(threshold: float = None):
    threshold = threshold or CFG["pseudo_threshold"]
    probs_arr = np.load(str(OUTPUT / "test_probs.npy"))
    files_arr = np.load(str(OUTPUT / "test_filenames.npy"), allow_pickle=True).tolist()

    max_probs  = probs_arr.max(axis=1)
    pred_lbls  = probs_arr.argmax(axis=1)
    mask       = max_probs >= threshold

    p_imgs = [files_arr[i] for i, m in enumerate(mask) if m]
    p_lbls = [int(pred_lbls[i]) for i, m in enumerate(mask) if m]

    pd.DataFrame({"IMAGE": p_imgs, "LABEL": p_lbls}).to_csv(
        str(OUTPUT / "pseudo_labels.csv"), index=False
    )

    print(f"[Pseudo-Label] threshold={threshold}")
    print(f"  Total test  : {len(files_arr):,}")
    print(f"  Accepted    : {len(p_imgs):,} ({len(p_imgs)/len(files_arr)*100:.1f}%)")
    print(f"  Rejected    : {len(files_arr)-len(p_imgs):,}")
    print(f"  Classes covered: {len(set(p_lbls))}/{CFG['num_classes']}")

    plt.figure(figsize=(8, 3))
    plt.hist(max_probs, bins=50, color='royalblue', edgecolor='white')
    plt.axvline(threshold, color='red', linestyle='--', label=f"threshold={threshold}")
    plt.title("Test-set Confidence Distribution")
    plt.xlabel("Max softmax probability")
    plt.legend(); plt.tight_layout(); plt.show()

    return p_imgs, p_lbls

pseudo_imgs, pseudo_labels = gen_pseudo_labels()


# ─────────────────────────────────────────────────────────────
# ── CELL 10 ── Retrain with pseudo-labels  (~20-25 min on T4)
# ─────────────────────────────────────────────────────────────

# Lower epoch counts for fine-tuning stage
_save = {k: CFG[k] for k in ("epochs", "warmup_epochs", "early_stop")}
CFG["epochs"]       = CFG["pseudo_epochs"]
CFG["warmup_epochs"] = 1
CFG["early_stop"]   = 5

print(f"Fine-tuning {CFG['epochs']} epochs with {len(pseudo_imgs):,} pseudo-labels...")
history2, best_acc2 = run_training(
    extra_images=pseudo_imgs,
    extra_labels=pseudo_labels,
    warm_start="best_model.pth",
)
print(f"\n🎯 Best val acc after pseudo-label fine-tune: {best_acc2:.2f}%")

# Restore cfg
for k, v in _save.items(): CFG[k] = v


# ─────────────────────────────────────────────────────────────
# ── CELL 11 ── Final inference → SUBMIT THIS FINAL.csv
# ─────────────────────────────────────────────────────────────

print("Generating final predictions with TTA ensemble...")
_, _ = infer_tta(tta_steps=CFG["tta_steps"], save_probs=False)

# Final summary
print("\n" + "=" * 60)
print("  🏆  SUBMISSION READY")
print("=" * 60)
df_final = pd.read_csv(str(OUTPUT / "FINAL.csv"))
print(df_final.head(8).to_string(index=False))
print(f"\n  Rows    : {len(df_final):,}")
print(f"  Classes : {df_final['LABEL'].nunique()} / {CFG['num_classes']}")
print(f"  Best val: {max(best_val_acc, best_acc2):.2f}%")
print("=" * 60)
print("\n  ✅  Go to /kaggle/working/ → download FINAL.csv → submit!")
