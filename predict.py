"""
predict.py — TTA inference + multi-model ensemble → generates FINAL.csv

How it works:
  1. For each trained backbone checkpoint, run TTA_STEPS augmented forward passes
  2. Average softmax probabilities across all TTA steps → per-model soft predictions
  3. Weighted average across all models (ENSEMBLE_WEIGHTS in config.py)
  4. argmax → final predicted class label
  5. Save to FINAL.csv

Usage:
    python predict.py           # uses all backbones in config.BACKBONES
    python predict.py --tta 10  # override number of TTA steps
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

import config
from dataset import get_test_loader
from model import build_model
from utils import load_checkpoint, set_seed


# ─────────────────────────────────────────────────────────────
# 1. Run TTA inference for one backbone
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_tta_inference(model, backbone_name: str, device, tta_steps: int):
    """
    Run TTA inference: average softmax probs over `tta_steps` augmented views.

    Returns:
        averaged_probs : np.ndarray of shape (N, NUM_CLASSES)
        all_filenames  : list of image filenames in test order
    """
    img_size = config.IMG_SIZE_MAP.get(backbone_name, config.DEFAULT_IMG_SIZE)
    model.eval()
    all_filenames = None
    accumulated   = None   # sum of softmax probs across TTA steps

    for tta_idx in range(tta_steps):
        test_loader, filenames = get_test_loader(img_size, tta_idx=tta_idx)

        if all_filenames is None:
            all_filenames = filenames
        if accumulated is None:
            accumulated = np.zeros((len(filenames), config.NUM_CLASSES), dtype=np.float64)

        step_probs = []
        for images, _ in tqdm(test_loader, desc=f"  TTA {tta_idx+1}/{tta_steps}", leave=False):
            images = images.to(device, non_blocking=True)
            with autocast(enabled=config.USE_AMP):
                logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            step_probs.append(probs)

        step_probs_arr = np.concatenate(step_probs, axis=0)   # (N, C)
        accumulated   += step_probs_arr

    averaged_probs = accumulated / tta_steps
    return averaged_probs, all_filenames


# ─────────────────────────────────────────────────────────────
# 2. Main ensemble predict
# ─────────────────────────────────────────────────────────────

def main(tta_steps: int = None):
    set_seed(config.SEED)
    device   = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    tta_steps = tta_steps or config.TTA_STEPS

    print(f"\n{'='*60}")
    print(f"  TTA Ensemble Inference")
    print(f"  Device: {device} | TTA steps: {tta_steps}")
    print(f"{'='*60}\n")

    # Validate ensemble weights
    assert len(config.ENSEMBLE_WEIGHTS) == len(config.BACKBONES), \
        "ENSEMBLE_WEIGHTS must have same length as BACKBONES"

    all_filenames        = None
    ensemble_probs       = None
    total_weight         = sum(config.ENSEMBLE_WEIGHTS)

    for backbone_name, weight in zip(config.BACKBONES, config.ENSEMBLE_WEIGHTS):
        safe_name = backbone_name.replace("/", "_").replace(".", "_")
        ckpt_path = os.path.join(config.OUTPUT_DIR, f"best_{safe_name}.pth")

        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path} — skipping.")
            total_weight -= weight
            continue

        print(f"\n[Model] Loading {backbone_name} (weight={weight})")
        model = build_model(backbone_name, pretrained=False).to(device)
        load_checkpoint(ckpt_path, model)

        probs, filenames = run_tta_inference(model, backbone_name, device, tta_steps)
        print(f"  → Inference done. probs shape: {probs.shape}")

        if all_filenames is None:
            all_filenames  = filenames
            ensemble_probs = weight * probs
        else:
            ensemble_probs += weight * probs

        del model
        torch.cuda.empty_cache()

    if ensemble_probs is None:
        raise RuntimeError("No valid checkpoints found! Please train first.")

    # Normalize by total (in case some models were skipped)
    ensemble_probs /= total_weight

    # Final predictions
    predictions = ensemble_probs.argmax(axis=1)
    max_confs   = ensemble_probs.max(axis=1)

    print(f"\n[Ensemble] Prediction confidence stats:")
    print(f"  Mean max prob: {max_confs.mean():.4f}")
    print(f"  Min  max prob: {max_confs.min():.4f}")
    print(f"  % predictions > 0.9 conf: {(max_confs > 0.9).mean()*100:.1f}%")

    # Save FINAL.csv
    final_path = os.path.join(config.BASE_DIR, "FINAL.csv")
    df = pd.DataFrame({"IMAGE": all_filenames, "LABEL": predictions.astype(int)})
    df.to_csv(final_path, index=False)
    print(f"\n✅ FINAL.csv saved → {final_path}")
    print(f"   Rows: {len(df)} | Unique classes predicted: {df['LABEL'].nunique()}")

    # Also save probabilities for pseudo-labelling
    np.save(os.path.join(config.OUTPUT_DIR, "test_probs.npy"), ensemble_probs)
    np.save(os.path.join(config.OUTPUT_DIR, "test_filenames.npy"), np.array(all_filenames))
    print(f"   Probabilities saved (for pseudo-labelling).")

    # Submission format sanity check
    test_df = pd.read_csv(config.TEST_CSV)
    assert len(df) == len(test_df), f"Row mismatch: got {len(df)}, expected {len(test_df)}"
    assert set(df.columns) == {"IMAGE", "LABEL"}, "Column names must be IMAGE and LABEL"
    assert df["LABEL"].between(0, config.NUM_CLASSES - 1).all(), "Label out of range!"
    print("   ✅ Submission format checks passed.")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA Ensemble Inference")
    parser.add_argument("--tta", type=int, default=None, help="Number of TTA steps (overrides config)")
    args = parser.parse_args()
    main(tta_steps=args.tta)
