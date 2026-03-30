# check_data.py
# This script checks your data file to understand its structure

import pandas as pd
import os

print("="*60)
print("CANCER DATA INSPECTION")
print("="*60)

# Check if file exists
file_path = "data/cancer_data.csv"

if not os.path.exists(file_path):
    print(f"ERROR: File not found at {file_path}")
    print("Please place your data file in the 'data' folder")
    exit()

# Get file size
file_size = os.path.getsize(file_path) / (1024**3)  # Convert to GB
print(f"\n1. File size: {file_size:.2f} GB")

# Read first 5 rows to see column names
print("\n2. Reading first 5 rows to understand structure...")
df_sample = pd.read_csv(file_path, nrows=5)

print(f"\n3. Number of columns: {len(df_sample.columns)}")
print(f"\n4. Column names:")
for i, col in enumerate(df_sample.columns, 1):
    print(f"   {i}. {col}")

print(f"\n5. First few rows of data:")
print(df_sample.head())

print("\n6. Data types:")
print(df_sample.dtypes)

print("\n7. Checking for target column...")
possible_targets = ['target', 'cancer', 'diagnosis', 'malignant', 'label']
target_col = None

for col in df_sample.columns:
    if col.lower() in possible_targets:
        target_col = col
        break

if target_col:
    print(f"   Found target column: {target_col}")
    print(f"   Values: {df_sample[target_col].unique()}")
else:
    print("   Could not identify target column automatically")
    print("   Please look at column names above and identify which column")
    print("   contains cancer diagnosis (0=benign, 1=malignant)")

print("\n" + "="*60)
print("INSPECTION COMPLETE")
print("="*60)
