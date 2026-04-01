
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import os
import matplotlib.pyplot as plt

print("="*60)
print("CANCER DETECTION PREDICTOR")
print("="*60)

# Load models
print("\n1. Loading models...")
feature_extractor = load_model('feature_extractor.h5')
with open('random_forest_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Cancer types (in order of your dataset)
cancer_types = [
    'ALL', 'Brain Cancer', 'Breast Cancer', 'Cervical Cancer',
    'Kidney Cancer', 'Lung and Colon Cancer', 'Lymphoma', 'Oral Cancer'
]

print(f"   ✓ Loaded model for {len(cancer_types)} cancer types")

def predict_cancer(image_path):
    """
    Predict cancer type from an image
    """
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Extract features
    features = feature_extractor.predict(img_array, verbose=0)
    
    # Predict
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3 = [(cancer_types[i], probabilities[i] * 100) for i in top_3_idx]
    
    return cancer_types[prediction], probabilities[prediction] * 100, top_3

def predict_batch(image_folder):
    """
    Predict for all images in a folder
    """
    results = []
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    print(f"\n   Found {len(image_files)} images")
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            cancer_type, confidence, _ = predict_cancer(img_path)
            results.append({
                'image': img_file,
                'predicted_cancer': cancer_type,
                'confidence': confidence
            })
            print(f"   ✓ {img_file}: {cancer_type} ({confidence:.1f}%)")
        except Exception as e:
            print(f"   ✗ {img_file}: Error - {e}")
    
    return results

# Main menu
print("\n2. Choose prediction mode:")
print("   1. Predict single image")
print("   2. Predict all images in a folder")
print("   3. Test with sample images")

choice = input("\nEnter choice (1/2/3): ")

if choice == '1':
    # Single image prediction
    image_path = input("\nEnter image path: ")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
    else:
        cancer_type, confidence, top_3 = predict_cancer(image_path)
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\n🎗️  Predicted Cancer: {cancer_type}")
        print(f"📊 Confidence: {confidence:.1f}%")
        
        print("\n📈 Top 3 Predictions:")
        for i, (c_type, conf) in enumerate(top_3, 1):
            bar = "█" * int(conf/5) + "░" * (20 - int(conf/5))
            print(f"   {i}. {c_type:<25} {bar} {conf:.1f}%")
        
        # Show image
        img = plt.imread(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {cancer_type}\nConfidence: {confidence:.1f}%")
        plt.axis('off')
        plt.show()

elif choice == '2':
    # Batch prediction
    folder_path = input("\nEnter folder path: ")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
    else:
        results = predict_batch(folder_path)
        
        # Save results to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('batch_predictions.csv', index=False)
        print(f"\n✅ Results saved to: batch_predictions.csv")
        
        # Summary
        print("\n📊 Summary:")
        summary = df['predicted_cancer'].value_counts()
        for cancer, count in summary.items():
            print(f"   {cancer}: {count} images")

elif choice == '3':
    # Test with sample from your dataset
    print("\n📁 Testing with sample images...")
    
    # Test one image from each cancer type
    dataset_path = r"C:\Ashish\test\dataset\Multi Cancer"
    
    for cancer in cancer_types:
        cancer_folder = os.path.join(dataset_path, cancer)
        if os.path.exists(cancer_folder):
            # Find first image in folder
            images = [f for f in os.listdir(cancer_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image = os.path.join(cancer_folder, images[0])
                pred_type, confidence, _ = predict_cancer(test_image)
                status = "✓" if pred_type == cancer else "✗"
                print(f"   {status} {cancer}: Predicted as {pred_type} ({confidence:.1f}%)")

print("\n" + "="*60)
print("PREDICTION COMPLETE!")
print("="*60)