# large_data_model.py
# Cancer prediction model for large dataset (9GB+)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LARGE DATA CANCER PREDICTION MODEL")
print("="*70)

# Configuration
FILE_PATH = "data/cancer_data.csv"
CHUNK_SIZE = 50000
TARGET_COL = None

print("\n1. Analyzing data structure...")

# Read first chunk
first_chunk = pd.read_csv(FILE_PATH, nrows=1000)

print(f"\n   Found {len(first_chunk.columns)} columns:")
cols_to_show = first_chunk.columns.tolist()[:10]
for i, col in enumerate(cols_to_show, 1):
    print(f"   {i}. {col}")
if len(first_chunk.columns) > 10:
    print(f"   ... and {len(first_chunk.columns) - 10} more columns")

# Identify target column
if TARGET_COL is None:
    print("\n   Common target names: target, cancer, diagnosis, malignant, label")
    TARGET_COL = input("   Enter your target column name: ")

print(f"\n   Using: {TARGET_COL}")

feature_cols = [col for col in first_chunk.columns if col != TARGET_COL]
print(f"   Features: {len(feature_cols)}")

# Check class distribution
print("\n2. Checking class distribution...")
class_counts = first_chunk[TARGET_COL].value_counts()
print(f"   Class 0 (Benign): {class_counts.get(0, 0)}")
print(f"   Class 1 (Malignant): {class_counts.get(1, 0)}")

if 1 in class_counts and class_counts[1] > 0:
    imbalance_ratio = class_counts.get(0, 0) / class_counts[1]
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
else:
    imbalance_ratio = 1

# XGBoost parameters
params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'scale_pos_weight': imbalance_ratio,
    'nthread': -1,
    'seed': 42
}

print("\n3. Training XGBoost on large dataset...")
print(f"   Chunk size: {CHUNK_SIZE:,} rows")

model = None
chunk_count = 0
total_rows = 0

chunk_iter = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)

for chunk in tqdm(chunk_iter, desc="   Processing"):
    chunk_count += 1
    total_rows += len(chunk)
    
    X_chunk = chunk[feature_cols]
    y_chunk = chunk[TARGET_COL]
    
    # Handle missing values
    X_chunk = X_chunk.fillna(X_chunk.median())
    
    dtrain = xgb.DMatrix(X_chunk, label=y_chunk)
    
    if model is None:
        model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
    else:
        model = xgb.train(params, dtrain, num_boost_round=50, 
                         xgb_model=model, verbose_eval=False)
    
    if chunk_count % 5 == 0:
        y_pred = model.predict(dtrain)
        auc = roc_auc_score(y_chunk, y_pred)
        print(f"\n   Chunk {chunk_count}: AUC={auc:.4f}")

print(f"\n   Total: {total_rows:,} rows, {chunk_count} chunks")

# Save model
print("\n4. Saving model...")
model.save_model('models/xgboost_cancer_model.json')
joblib.dump(feature_cols, 'models/feature_names.pkl')
print("   Saved to models/xgboost_cancer_model.json")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
