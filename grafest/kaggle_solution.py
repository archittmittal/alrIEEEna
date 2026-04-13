import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("imblearn not found. SMOTE will be skipped (pip install imbalanced-learn)")
    SMOTE = None

import warnings
warnings.filterwarnings('ignore')

#############################################
# 1. Data Understanding & Configuration
#############################################
print("--- 1. Data Understanding & Configuration ---")
DATA_DIR = "./"  # Update path if needed
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "tier3.csv") # Using tier3 as the main test source
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV) # Tier 3 has image_id and label

print(f"Train dataset shape: {train_df.shape}")
print(f"Test (Tier 3) dataset shape: {test_df.shape}")
print(f"Classes: {train_df['label'].nunique()} unique labels.")

#############################################
# 2. Feature Extraction Function
#############################################
def extract_image_features(img_name, img_dir, resize_dim=(16, 16)):
    """
    Extracts tabular features from an image to use with XGBoost/LightGBM.
    We extract RGB & HSV stats, histograms, and flattened pixels to maximize accuracy.
    """
    img_path = os.path.join(img_dir, img_name)
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return np.zeros(resize_dim[0] * resize_dim[1] * 3 + 24 + 48)
        
    img_arr = np.array(img)
    img_hsv = np.array(img.convert('HSV'))
    
    # Feature set 1: Channel statistics (Mean, Std, Min, Max) for RGB and HSV
    stats = []
    for i in range(3):
        # RGB
        channel = img_arr[:, :, i]
        stats.extend([channel.mean(), channel.std(), channel.min(), channel.max()])
        # HSV
        hsv_chan = img_hsv[:, :, i]
        stats.extend([hsv_chan.mean(), hsv_chan.std(), hsv_chan.min(), hsv_chan.max()])
        
    # Feature set 2: Color Histograms (16 bins per channel) for RGB
    hist_features = []
    for i in range(3):
        hist, _ = np.histogram(img_arr[:, :, i], bins=16, range=(0, 256))
        hist_features.extend(hist)
        
    # Feature set 3: Downsampled flattened pixels
    img_resized = img.resize(resize_dim)
    pixels = np.array(img_resized).flatten()
    
    # Combine all
    features = np.concatenate([stats, hist_features, pixels])
    return features

print("Extracting features from training images... (this may take a minute)")
X_train_raw = np.array([extract_image_features(img, TRAIN_IMG_DIR) for img in train_df['image_id']])
y_train_raw = train_df['label'].values

print("Extracting features from Tier 3 test images...")
X_test_raw = np.array([extract_image_features(img, TEST_IMG_DIR) for img in test_df['image_id']])
y_test_raw = test_df['label'].values

#############################################
# 3. Data Preprocessing & SMOTE
#############################################
print("\n--- 3. Preprocessing ---")
label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train_raw)
y_test_enc = label_enc.transform(y_test_raw)

# Splitting train further into train and validation for early stopping
X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_enc, test_size=0.15, random_state=42, stratify=y_train_enc)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_raw)

# Handling Imbalance
if SMOTE is not None:
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
else:
    X_train_bal, y_train_bal = X_train_scaled, y_train

print(f"Training set shape after SMOTE: {X_train_bal.shape}")

#############################################
# 4. Model Training (with Epoch Accuracies)
#############################################
print("\n--- 4. Model Selection & Tuning (XGBoost) ---")
# We use LightGBM or XGBoost - XGBoost makes it easy to track "epochs" (n_estimators) explicitly.
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric=['merror', 'mlogloss'],
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1
)

# Fit the model: watch validation accuracy over epochs
print("Starting Training (evaluating per epoch/round)...")
xgb_model.fit(
    X_train_bal, y_train_bal,
    eval_set=[(X_val_scaled, y_val), (X_train_bal, y_train_bal)],
    verbose=10
)

# If you specifically want to see exact Accuracy rather than merror:
results = xgb_model.evals_result()
val_merror = results['validation_0']['merror']
print("\nAccuracy sample from the last 5 epochs:")
for i, err in enumerate(val_merror[-5:]):
    print(f"Epoch {100 - 5 + i}: Validation Accuracy: {1.0 - err:.4f}")

#############################################
# 5. Final Evaluation on Tier 3 Test Data
#############################################
print("\n--- 5. Evaluation on Tier 3 (Main) ---")
# Predict on tier3 target set
y_pred_enc = xgb_model.predict(X_test_scaled)

acc = accuracy_score(y_test_enc, y_pred_enc)
prec = precision_score(y_test_enc, y_pred_enc, average='macro')
rec = recall_score(y_test_enc, y_pred_enc, average='macro')
f1 = f1_score(y_test_enc, y_pred_enc, average='macro')

print(f"TIER 3 FINAL METRICS:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n--- Classification Report (Tier 3) ---")
print(classification_report(y_test_enc, y_pred_enc, target_names=label_enc.classes_))

# Plot confusion matrix
cm = confusion_matrix(y_test_enc, y_pred_enc)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, cmap='Blues')
plt.title('Confusion Matrix - Tier 3 Test Data')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_tier3.png')
print("Saved confusion matrix as 'confusion_matrix_tier3.png'")

# Output predictions to CSV if needed
submission = pd.DataFrame({
    'image_id': test_df['image_id'],
    'actual_label': test_df['label'],
    'predicted_label': label_enc.inverse_transform(y_pred_enc)
})
submission.to_csv('tier3_eval_results.csv', index=False)
print("Saved detailed prediction comparisons to 'tier3_eval_results.csv'")
