import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

print("="*70)
print("TEST ON SAMPLE - Quick Validation")
print("="*70)

DATASET_PATH = r"C:\Ashish\test\dataset\Multi Cancer"
MAX_IMAGES_PER_CLASS = 50  # Only 50 images per cancer type

print("\n1. Loading sample images (only 50 per type)...")
images = []
labels = []
class_names = []

cancer_folders = [f for f in os.listdir(DATASET_PATH) 
                  if os.path.isdir(os.path.join(DATASET_PATH, f))]

for class_idx, cancer in enumerate(cancer_folders):
    class_names.append(cancer)
    cancer_path = os.path.join(DATASET_PATH, cancer)
    
    # Get image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(cancer_path).glob(ext)))
    
    # Take only first MAX_IMAGES_PER_CLASS
    for img_path in image_files[:MAX_IMAGES_PER_CLASS]:
        try:
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            images.append(img)
            labels.append(class_idx)
        except:
            continue
    
    print(f"   {cancer}: {len(images)} images loaded")

X = np.array(images)
y = np.array(labels)

print(f"\n📊 Total images loaded: {len(X)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n2. Building simple model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n3. Training (quick test)...")
history = model.fit(X_train, y_train, 
                    epochs=5, 
                    validation_split=0.2,
                    verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

print("\n📊 This is just a test! Results may improve with more data.")
