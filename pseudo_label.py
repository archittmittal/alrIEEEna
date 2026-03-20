"""
pseudo_label.py — Semi-supervised pseudo-labeling of the test set.

How it works:
  1. Load test probabilities saved by predict.py (test_probs.npy)
  2. Keep only high-confidence predictions (max softmax >= PSEUDO_CONF_THRESHOLD)
  3. Save as outputs/pseudo_labels.csv (IMAGE, LABEL)
  4. Re-train with: python train.py --pseudo

This leverages the 24,858 unlabeled test images as extra training data —
a massive edge over teams that ignore the test distribution entirely.

Usage:
    python pseudo_label.py                  # uses default threshold from config
    python pseudo_label.py --threshold 0.95 # stricter threshold
"""

import argparse
import os
import numpy as np
import pandas as pd

import config


def generate_pseudo_labels(threshold: float = None):
    threshold = threshold or config.PSEUDO_CONF_THRESHOLD

    probs_path = os.path.join(config.OUTPUT_DIR, "test_probs.npy")
    names_path = os.path.join(config.OUTPUT_DIR, "test_filenames.npy")

    if not os.path.exists(probs_path) or not os.path.exists(names_path):
        raise FileNotFoundError(
            "test_probs.npy or test_filenames.npy not found. "
            "Please run predict.py first."
        )

    probs     = np.load(probs_path)                     # (N, 397)
    filenames = np.load(names_path, allow_pickle=True)  # (N,)

    max_probs  = probs.max(axis=1)
    pred_labels = probs.argmax(axis=1)

    # Keep high-confidence predictions
    mask = max_probs >= threshold

    pseudo_images = filenames[mask].tolist()
    pseudo_labels = pred_labels[mask].tolist()

    print(f"\n[Pseudo-Label] Threshold: {threshold:.2f}")
    print(f"  Total test samples:         {len(filenames)}")
    print(f"  High-confidence (>={threshold:.2f}): {mask.sum()} ({mask.mean()*100:.1f}%)")
    print(f"  Dropped (low confidence):   {(~mask).sum()}")

    # Save
    pseudo_path = os.path.join(config.OUTPUT_DIR, "pseudo_labels.csv")
    df = pd.DataFrame({"IMAGE": pseudo_images, "LABEL": pseudo_labels})
    df.to_csv(pseudo_path, index=False)

    print(f"\n  ✅ Saved {len(df)} pseudo-labels → {pseudo_path}")
    print(f"  Unique classes covered: {df['LABEL'].nunique()} / {config.NUM_CLASSES}")

    # Distribution check
    class_counts = df['LABEL'].value_counts()
    print(f"  Most common pseudo-class: {class_counts.idxmax()} ({class_counts.max()} samples)")
    print(f"  Least common pseudo-class: {class_counts.idxmin()} ({class_counts.min()} samples)")
    print(f"\n  ▶  Now run: python train.py --pseudo")
    return pseudo_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from test predictions")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Min softmax confidence to accept a pseudo-label (default from config)")
    args = parser.parse_args()
    generate_pseudo_labels(threshold=args.threshold)
