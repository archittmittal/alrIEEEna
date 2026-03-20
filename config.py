"""
config.py — Central configuration for the IEEE GEHU hackathon ML pipeline.
Edit these values to control all aspects of training and inference.
"""

import os

# ─────────────── Paths ───────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV   = os.path.join(BASE_DIR, "TRAIN.csv")
TEST_CSV    = os.path.join(BASE_DIR, "TEST.csv")
IMG_DIR     = os.path.join(BASE_DIR, "images")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────── Dataset ───────────────
NUM_CLASSES     = 397
VAL_SPLIT       = 0.15          # 15% held-out for validation
SEED            = 42
LABEL_SMOOTHING = 0.1

# ─────────────── Image / Augmentation ───────────────
# Use 384 for EfficientNet/ConvNeXt, 448 for EVA-02
IMG_SIZE_MAP = {
    "efficientnetv2_l.in21k": 384,
    "convnext_base.fb_in22k_ft_in1k": 384,
    "eva02_base_patch14_448.mim_in22k_ft_in22k_ft_in1k": 448,
}
DEFAULT_IMG_SIZE = 384          # fallback

# ─────────────── Training ───────────────
BATCH_SIZE      = 16            # reduce to 8 if GPU OOM
NUM_WORKERS     = 4
PIN_MEMORY      = True
EPOCHS          = 30
LR              = 3e-4
MIN_LR          = 1e-6
WEIGHT_DECAY    = 1e-2
GRAD_CLIP       = 5.0
WARMUP_EPOCHS   = 2
EARLY_STOP_PATIENCE = 8         # epochs without improvement → stop
MIXUP_ALPHA     = 0.4           # 0 = disable mixup
CUTMIX_ALPHA    = 0.4           # 0 = disable cutmix
USE_AMP         = True          # mixed precision (requires CUDA)

# ─────────────── Backbones to train (in order) ───────────────
# timm model names — they will be downloaded automatically
BACKBONES = [
    "efficientnetv2_l.in21k",
    "convnext_base.fb_in22k_ft_in1k",
    # Uncomment if you have strong GPU (≥20 GB VRAM):
    # "eva02_base_patch14_448.mim_in22k_ft_in22k_ft_in1k",
]

# ─────────────── Ensemble weights for each backbone ───────────────
# Must match order and length of BACKBONES above
ENSEMBLE_WEIGHTS = [0.55, 0.45]  # [eff_l, convnext]

# ─────────────── TTA ───────────────
TTA_STEPS = 8    # number of augmented views at inference (more = slower but better)

# ─────────────── Pseudo-labeling ───────────────
PSEUDO_CONF_THRESHOLD = 0.92    # only keep predictions with max softmax >= this
PSEUDO_EPOCHS         = 10      # fine-tune epochs after adding pseudo-labels

# ─────────────── Misc ───────────────
DEVICE = "cuda"   # "cpu" | "cuda" | "mps" (Apple Silicon)
