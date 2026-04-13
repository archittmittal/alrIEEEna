import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION - TARGET: 90%+ ACCURACY
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # Increased for deep learning convergence
MAX_LR = 2e-4 # Better stability for fine-tuning
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = "/kaggle/input/datasets/archittmittal/grafestt"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "tier3.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test_images")

# ============================================================
# DATASET
# ============================================================
class ImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image

        label = int(self.df.iloc[idx, 1])
        return image, label

# ============================================================
# AUGMENTATION - HIGH PERFORMANCE SATELLITE SUITE
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=9), # Advanced augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ============================================================
# TRAIN & VALIDATION FUNCTIONS
# ============================================================
def train():
    print(f"🚀 High-Accuracy 90%+ Pipeline | Device: {DEVICE}")
    
    # --- Load Data & Encode ---
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(TEST_CSV)

    label_enc = LabelEncoder()
    train_df['label'] = label_enc.fit_transform(train_df['label'])
    
    has_val_labels = 'label' in val_df.columns
    if has_val_labels:
        val_df['label'] = label_enc.transform(val_df['label'])

    # --- Dataloaders ---
    train_ds = ImageDataset(train_df, TRAIN_IMG_DIR, train_transform)
    val_ds   = ImageDataset(val_df, TEST_IMG_DIR, val_transform, is_test=not has_val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model: EfficientNet-B0 ---
    model = models.efficientnet_b0(weights='DEFAULT')
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=EPOCHS)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}
    best_val_acc = 0.0

    # --- MAIN LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{train_loss/(pbar.n+1):.4f}", 'acc': f"{100*train_correct/train_total:.2f}%"})

        if has_val_labels:
            model.eval()
            val_correct, val_total = 0, 0
            all_v_preds, all_v_labels = [], []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_v_preds.extend(predicted.cpu().numpy())
                    all_v_labels.extend(labels.cpu().numpy())
            
            val_acc = 100 * val_correct / val_total
            macro_f1 = f1_score(all_v_labels, all_v_preds, average='macro')
            print(f"⭐ Epoch {epoch+1} | Val Acc: {val_acc:.2f}% | F1: {macro_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"🔥 Best Model Updated!")

    # --- INFERENCE WITH 5-PASS TTA ---
    print("\n🏁 FINAL INFERENCE: 5-Pass TTA...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_df = pd.read_csv(TEST_CSV)
    
    tta_transforms = [
        val_transform,
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((90, 90)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((270, 270)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ]
    
    all_tta_logits = []
    for t_idx, trans in enumerate(tta_transforms):
        print(f"  TTA Pass {t_idx+1}/5")
        test_ds = ImageDataset(test_df, TEST_IMG_DIR, trans, is_test=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        logits = []
        with torch.no_grad():
            for images in tqdm(test_loader):
                outputs = model(images.to(DEVICE))
                logits.append(outputs.cpu())
        all_tta_logits.append(torch.cat(logits))
    
    final_logits = torch.stack(all_tta_logits).mean(dim=0)
    _, final_preds = torch.max(final_logits, 1)
    
    submission_df = pd.DataFrame({
        'IMAGE': test_df['image_id'], 
        'LABEL': label_enc.inverse_transform(final_preds.numpy())
    })
    submission_df.to_csv('submission.csv', index=False)
    print(f"✅ SUBMISSION READY: Best Val Acc was {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()
