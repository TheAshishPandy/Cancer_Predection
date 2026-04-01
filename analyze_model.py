
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*60)
print("MODEL PERFORMANCE ANALYSIS")
print("="*60)

DATASET_PATH = r"C:\Ashish\test\dataset\Multi Cancer"

# Load model
print("\n1. Loading model...")
feature_extractor = load_model('feature_extractor.h5')
with open('random_forest_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Load validation data
print("\n2. Loading validation data...")
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(val_generator.class_indices.keys())

# Extract features and predict
print("\n3. Making predictions...")
X_val = []
y_val = []
y_pred = []

for i in range(len(val_generator)):
    batch_x, batch_y = val_generator[i]
    features = feature_extractor.predict(batch_x, verbose=0)
    pred = classifier.predict(features)
    
    X_val.append(features)
    y_val.append(batch_y)
    y_pred.extend(pred)

y_val_labels = np.argmax(np.vstack(y_val), axis=1)

# Create confusion matrix
print("\n4. Creating confusion matrix...")
cm = confusion_matrix(y_val_labels, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Cancer Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("   ✓ Saved: confusion_matrix.png")

# Find misclassifications
print("\n5. Analyzing misclassifications...")
misclassifications = []

for i, (true, pred) in enumerate(zip(y_val_labels, y_pred)):
    if true != pred:
        misclassifications.append({
            'true_label': class_names[true],
            'predicted_label': class_names[pred]
        })

df_mis = pd.DataFrame(misclassifications)
print("\n   Most common misclassifications:")
print(df_mis.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False).head(10))

# Per-class performance
print("\n6. Per-class performance:")
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_val_labels, y_pred, average=None)
recall = recall_score(y_val_labels, y_pred, average=None)
f1 = f1_score(y_val_labels, y_pred, average=None)

perf_df = pd.DataFrame({
    'Cancer Type': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}).round(4)

print(perf_df.to_string(index=False))

# Recommendations
print("\n7. Recommendations for improvement:")
for i, row in perf_df.iterrows():
    if row['F1-Score'] < 0.7:
        print(f"   🔧 Need more data for: {row['Cancer Type']}")
        print(f"      Current F1: {row['F1-Score']:.3f}")

print("\n" + "="*60)