
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("IMPROVING CANCER DETECTION MODEL")
print("="*60)

DATASET_PATH = r"C:\Ashish\test\dataset\Multi Cancer"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load your existing feature extractor
print("\n1. Loading feature extractor...")
feature_extractor = load_model('feature_extractor.h5')
print("   ✓ Loaded")

# Load training data again with augmentation
print("\n2. Loading data with augmentation for weak classes...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Extract features with augmentation
print("\n3. Extracting features with augmentation...")
X_train = []
y_train = []

for i in range(len(train_generator)):
    batch_x, batch_y = train_generator[i]
    features = feature_extractor.predict(batch_x, verbose=0)
    X_train.append(features)
    y_train.append(batch_y)
    
    if i % 100 == 0:
        print(f"   Processed {i}/{len(train_generator)} batches")

X_train = np.vstack(X_train)
y_train_labels = np.argmax(np.vstack(y_train), axis=1)

# Train improved classifier
print("\n4. Training improved classifier...")
rf_improved = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=20,      # Deeper trees
    random_state=42,
    n_jobs=-1
)
rf_improved.fit(X_train, y_train_labels)

# Evaluate on validation
print("\n5. Evaluating on validation set...")
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

X_val = []
y_val = []

for i in range(len(val_generator)):
    batch_x, batch_y = val_generator[i]
    features = feature_extractor.predict(batch_x, verbose=0)
    X_val.append(features)
    y_val.append(batch_y)

X_val = np.vstack(X_val)
y_val_labels = np.argmax(np.vstack(y_val), axis=1)

y_pred = rf_improved.predict(X_val)
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_val_labels, y_pred)
print(f"\n✅ Improved Accuracy: {accuracy:.4f}")

class_names = list(val_generator.class_indices.keys())
print("\n📊 Improved Classification Report:")
print(classification_report(y_val_labels, y_pred, target_names=class_names))

# Save improved model
print("\n6. Saving improved model...")
with open('random_forest_classifier_improved.pkl', 'wb') as f:
    pickle.dump(rf_improved, f)

print("   ✓ Improved classifier saved")

print("\n" + "="*60)
print("IMPROVEMENT COMPLETE!")
print("="*60)