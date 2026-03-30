# predict.py
# Make predictions on new patients

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

print("="*60)
print("CANCER PREDICTION - MAKE PREDICTIONS")
print("="*60)

# Load model
print("\n1. Loading model...")

try:
    model = xgb.Booster()
    model.load_model('models/xgboost_cancer_model.json')
    feature_names = joblib.load('models/feature_names.pkl')
    print(f"   Loaded model with {len(feature_names)} features")
except:
    print("   Model not found! Training model first...")
    print("   Run: python large_data_model.py")
    exit()

# Get patient data
print("\n2. Enter patient data:")
print("   Option 1: Single patient (manual entry)")
print("   Option 2: Batch from CSV file")

choice = input("   Choose (1 or 2): ")

if choice == '1':
    patient_data = {}
    print("\n   Enter values (press Enter for average):")
    
    # Try to load averages
    avg_values = {}
    if os.path.exists('data/cancer_data.csv'):
        sample = pd.read_csv('data/cancer_data.csv', nrows=1000)
        for col in feature_names:
            if col in sample.columns:
                avg_values[col] = sample[col].mean()
    
    # Ask for first 5 important features
    for feature in feature_names[:5]:
        default = avg_values.get(feature, 0)
        val = input(f"   {feature} (default={default:.2f}): ")
        patient_data[feature] = float(val) if val else default
    
    # Use averages for others
    for feature in feature_names[5:]:
        patient_data[feature] = avg_values.get(feature, 0)
    
    patient_df = pd.DataFrame([patient_data])
    
else:
    file_path = input("   Enter CSV file path: ")
    if not os.path.exists(file_path):
        print(f"   File not found!")
        exit()
    patient_df = pd.read_csv(file_path)
    print(f"   Loaded {len(patient_df)} patients")

# Prepare data
print("\n3. Making predictions...")
for f in feature_names:
    if f not in patient_df.columns:
        patient_df[f] = 0

patient_df = patient_df[feature_names]
patient_df = patient_df.fillna(patient_df.median())

dmatrix = xgb.DMatrix(patient_df)
predictions = model.predict(dmatrix)

# Show results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

for i, prob in enumerate(predictions):
    risk = prob * 100
    pred = "MALIGNANT (Cancer)" if prob > 0.5 else "BENIGN (No Cancer)"
    
    print(f"\nPatient {i+1}:")
    print(f"   Risk: {risk:.1f}%")
    print(f"   Prediction: {pred}")
    
    if risk > 70:
        print("   ⚠️  HIGH RISK - Urgent care needed")
    elif risk > 50:
        print("   ⚠️  ELEVATED RISK - Further testing")
    elif risk > 30:
        print("   ℹ️  MODERATE RISK - Monitor")
    else:
        print("   ✅  LOW RISK - Routine checkup")

# Save results
save = input("\nSave results? (y/n): ")
if save.lower() == 'y':
    results = pd.DataFrame({
        'patient': range(1, len(predictions)+1),
        'cancer_probability': predictions,
        'prediction': ['Malignant' if p>0.5 else 'Benign' for p in predictions]
    })
    results.to_csv('predictions_results.csv', index=False)
    print("   Saved to predictions_results.csv")
