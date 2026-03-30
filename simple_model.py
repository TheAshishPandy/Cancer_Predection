# simple_model.py
# A simple cancer prediction model to test your data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SIMPLE CANCER PREDICTION MODEL")
print("="*60)

# Step 1: Load data (only first 10,000 rows for quick testing)
print("\n1. Loading data (first 10,000 rows)...")
df = pd.read_csv("data/cancer_data.csv", nrows=10000)

print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

# Step 2: Identify target column
print("\n2. Identifying target column...")

target_candidates = ['target', 'cancer', 'diagnosis', 'malignant', 'label', 'class']

target_col = None
for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("   Please enter the target column name from the list below:")
    print(f"   Columns: {df.columns.tolist()}")
    target_col = input("   Target column name: ")

print(f"   Using target column: {target_col}")

# Step 3: Prepare features (X) and target (y)
print("\n3. Preparing data...")
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"   Target values: {y.unique()}")
print(f"   Class distribution:\n{y.value_counts()}")

# Step 4: Split into train and test
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training: {len(X_train)} rows, Testing: {len(X_test)} rows")

# Step 5: Train model
print("\n5. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("   Training complete!")

# Step 6: Make predictions
print("\n6. Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.4f}")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred))

print("\n   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Feature importance
print("\n7. Top 10 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Step 8: Save model
print("\n8. Saving model...")
import joblib
joblib.dump(model, 'models/simple_model.pkl')
print("   Model saved to models/simple_model.pkl")

print("\n" + "="*60)
print("SIMPLE MODEL COMPLETE!")
print("="*60)
