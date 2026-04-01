# Create fast_image_model.py

import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FAST CANCER DETECTION - Feature Extraction Method")
print("="*70)

# Configuration
DATASET_PATH = r"C:\Ashish\test\dataset\Multi Cancer"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Step 1: Load data using smaller image size
print("\n1. Loading images with reduced size (faster)...")
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=False  # Important for feature extraction
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class names
class_names = list(train_generator.class_indices.keys())
print(f"\n🎗️  Cancer Types: {len(class_names)}")
for i, name in enumerate(class_names):
    print(f"   {i}: {name}")

# Step 2: Extract features using pre-trained model (without training)
print("\n2. Extracting features from images (this is fast!)...")

# Load pre-trained ResNet50 without top layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Extract features from training images
print("   Extracting training features...")
X_train = []
y_train = []

for i in tqdm(range(len(train_generator))):
    batch_x, batch_y = train_generator[i]
    features = base_model.predict(batch_x, verbose=0)
    X_train.append(features)
    y_train.append(batch_y)

X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
y_train_labels = np.argmax(y_train, axis=1)

print(f"   Training features shape: {X_train.shape}")

# Extract features from validation images
print("\n   Extracting validation features...")
X_val = []
y_val = []

for i in tqdm(range(len(val_generator))):
    batch_x, batch_y = val_generator[i]
    features = base_model.predict(batch_x, verbose=0)
    X_val.append(features)
    y_val.append(batch_y)

X_val = np.vstack(X_val)
y_val = np.vstack(y_val)
y_val_labels = np.argmax(y_val, axis=1)

print(f"   Validation features shape: {X_val.shape}")

# Step 3: Train a simple classifier on the features
print("\n3. Training Random Forest classifier on extracted features...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train_labels)

# Step 4: Evaluate
print("\n4. Evaluating model...")
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val_labels, y_pred)

print(f"\n✅ Validation Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_val_labels, y_pred, target_names=class_names))

# Step 5: Save models
print("\n5. Saving models...")
# Save the feature extractor and classifier
base_model.save('feature_extractor.h5')
with open('random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("   ✓ Model saved to: feature_extractor.h5")
print("   ✓ Classifier saved to: random_forest_classifier.pkl")

print("\n" + "="*70)
print("✅ FAST TRAINING COMPLETE!")
print("="*70)
print("\n⏱️  This should take minutes, not hours!")
