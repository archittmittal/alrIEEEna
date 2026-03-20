"""
dataset.py — Custom PyTorch Dataset with Albumentations augmentations.
Handles variable-size images, train/val split, and pseudo-labeled data merging.
"""

import os
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2

import config


# ─────────────────────────────────────────────────────────────
# 1. Albumentations transforms
# ─────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int):
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.65, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        A.RandomGridShuffle(grid=(3, 3), p=0.1),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        A.CoarseDropout(
            max_holes=8, max_height=img_size // 8, max_width=img_size // 8,
            min_holes=1, fill_value=0, p=0.3
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int):
    return A.Compose([
        A.Resize(height=int(img_size * 1.1), width=int(img_size * 1.1)),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int, tta_idx: int):
    """
    Returns one of several TTA augmentation pipelines.
    tta_idx cycles through 0..TTA_STEPS-1.
    """
    base = [
        A.Resize(height=int(img_size * 1.1), width=int(img_size * 1.1)),
        A.CenterCrop(height=img_size, width=img_size),
    ]
    tta_options = [
        [],                                              # 0: clean center crop
        [A.HorizontalFlip(p=1.0)],                      # 1: h-flip
        [A.VerticalFlip(p=1.0)],                        # 2: v-flip
        [A.RandomCrop(height=img_size, width=img_size)],# 3: random crop
        [A.Transpose(p=1.0)],                           # 4: transpose
        [A.Rotate(limit=15, p=1.0)],                    # 5: slight rotate
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],  # 6: both flips
        [A.Rotate(limit=30, p=1.0)],                    # 7: more rotate
    ]
    extra = tta_options[tta_idx % len(tta_options)]
    return A.Compose(
        base + extra + [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ─────────────────────────────────────────────────────────────
# 2. Dataset class
# ─────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """
    Generic dataset for both training (labelled) and test (unlabelled) images.

    Args:
        img_names   : list of image filenames (e.g. ['000001.jpg', ...])
        labels      : list of int labels (same length as img_names),
                      or None for test data
        img_dir     : directory containing all images
        transform   : albumentations Compose pipeline
    """

    def __init__(self, img_names: list, labels, img_dir: str, transform):
        self.img_names = img_names
        self.labels    = labels        # None for test
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.img_names[idx])
        # Always load as RGB — handles grayscale images too
        img = cv2.imread(path)
        if img is None:
            # Fallback to PIL for problematic files
            img = np.array(Image.open(path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        image_tensor = augmented["image"]          # float32 CHW

        if self.labels is not None:
            return image_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, self.img_names[idx]   # return filename for test


# ─────────────────────────────────────────────────────────────
# 3. Data loading helpers
# ─────────────────────────────────────────────────────────────

def load_csv(path: str):
    """Read a CSV into (list_of_image_names, list_of_labels or None)."""
    df = pd.read_csv(path)
    images = df["IMAGE"].tolist()
    labels = df["LABEL"].astype(int).tolist() if "LABEL" in df.columns else None
    return images, labels


def get_train_val_loaders(img_size: int, extra_images=None, extra_labels=None):
    """
    Build stratified train / validation DataLoaders.

    Args:
        img_size      : target square image size
        extra_images  : list of pseudo-labelled filenames (optional)
        extra_labels  : list of pseudo-labels (optional)
    Returns:
        train_loader, val_loader, train_labels (for class weight computation)
    """
    images, labels = load_csv(config.TRAIN_CSV)

    # Merge pseudo-labels if provided
    if extra_images and extra_labels:
        images = images + extra_images
        labels = labels + extra_labels
        print(f"[Dataset] Added {len(extra_images)} pseudo-labelled samples. "
              f"Total: {len(images)}")

    # Stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.VAL_SPLIT, random_state=config.SEED
    )
    train_idx, val_idx = next(sss.split(images, labels))

    train_imgs   = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_imgs     = [images[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    print(f"[Dataset] Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    train_dataset = ImageDataset(train_imgs, train_labels, config.IMG_DIR, get_train_transforms(img_size))
    val_dataset   = ImageDataset(val_imgs,   val_labels,   config.IMG_DIR, get_val_transforms(img_size))

    # Weighted sampler to oversample minority classes
    class_count = [0] * config.NUM_CLASSES
    for lbl in train_labels:
        class_count[lbl] += 1
    sample_weights = [1.0 / max(class_count[lbl], 1) for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_labels),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return train_loader, val_loader, train_labels


def get_test_loader(img_size: int, tta_idx: int = 0):
    """
    Build a DataLoader for the test set using one TTA transform variant.
    """
    images, _ = load_csv(config.TEST_CSV)
    test_dataset = ImageDataset(
        images, labels=None, img_dir=config.IMG_DIR,
        transform=get_tta_transforms(img_size, tta_idx)
    )
    loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return loader, images
