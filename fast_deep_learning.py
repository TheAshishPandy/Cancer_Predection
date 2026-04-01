import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2  # Smaller than ResNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("="*70)
print("FAST DEEP LEARNING - MobileNetV2")
print("="*70)

DATASET_PATH = r"C:\Ashish\test\dataset\Multi Cancer"
IMG_SIZE = (128, 128)  # Smaller images = faster training
BATCH_SIZE = 64  # Larger batch = faster processing

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Use MobileNetV2 (much smaller and faster than ResNet50)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n🧠 Model Summary:")
model.summary()

# Train with early stopping (will stop when no improvement)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\n🚀 Training (will stop automatically when done)...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Only 10 epochs max
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n✅ Validation Accuracy: {val_acc:.4f}")

# Save
model.save('fast_cancer_model.h5')
print("💾 Model saved to: fast_cancer_model.h5")