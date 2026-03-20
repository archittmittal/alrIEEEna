#!/usr/bin/env bash
# run.sh — One-shot pipeline runner for the IEEE GEHU hackathon
# Usage: bash run.sh
# Adjust BACKBONE and TTA as needed before running.

set -euo pipefail

echo "=========================================================="
echo "  IEEE GEHU alrIEEEna26 — Full ML Pipeline"
echo "=========================================================="

# ── STEP 1: Install dependencies ──────────────────────────────
echo ""
echo "[1/6] Installing requirements..."
pip install -q -r requirements.txt

# ── STEP 2: Train backbone 1 (EfficientNetV2-L) ───────────────
echo ""
echo "[2/6] Training EfficientNetV2-L..."
python train.py --backbone efficientnetv2_l.in21k

# ── STEP 3: Train backbone 2 (ConvNeXt-Base IN22K) ────────────
echo ""
echo "[3/6] Training ConvNeXt-Base..."
python train.py --backbone convnext_base.fb_in22k_ft_in1k

# ── STEP 4: Generate FINAL.csv (initial ensemble with TTA) ────
echo ""
echo "[4/6] Running TTA ensemble inference → FINAL.csv"
python predict.py

# ── STEP 5: Generate pseudo-labels from test predictions ───────
echo ""
echo "[5/6] Generating pseudo-labels..."
python pseudo_label.py --threshold 0.92

# ── STEP 6: Retrain with pseudo-labels and regenerate FINAL.csv
echo ""
echo "[6/6] Re-training with pseudo-labels..."
python train.py --pseudo
python predict.py

echo ""
echo "=========================================================="
echo "  ✅ All done! Submit: FINAL.csv"
echo "=========================================================="
