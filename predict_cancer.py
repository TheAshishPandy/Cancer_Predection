import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CANCER DETECTION PREDICTOR")
print("="*60)

# Load models
print("\n1. Loading models...")
feature_extractor = load_model('feature_extractor.h5', compile=False)
with open('random_forest_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Cancer types in order
cancer_types = [
    'ALL (Leukemia)', 
    'Brain Cancer', 
    'Breast Cancer', 
    'Cervical Cancer',
    'Kidney Cancer', 
    'Lung and Colon Cancer', 
    'Lymphoma', 
    'Oral Cancer'
]

print(f"   ✓ Loaded model for {len(cancer_types)} cancer types")

def predict_cancer(image_path):
    """Predict cancer type from an image"""
    try:
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
    except Exception as e:
        return None, 0, []

# Main menu
print("\n2. Choose prediction mode:")
print("   1. Predict single image")
print("   2. Predict all images in a folder")
print("   3. Test with sample images from dataset")

choice = input("\nEnter choice (1/2/3): ")

if choice == '1':
    # Single image prediction
    image_path = input("\nEnter image path: ")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
    else:
        cancer_type, confidence, top_3 = predict_cancer(image_path)
        
        if cancer_type:
            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            print(f"\n🎗️  Predicted Cancer: {cancer_type}")
            print(f"📊 Confidence: {confidence:.1f}%")
            
            print("\n📈 Top 3 Predictions:")
            for i, (c_type, conf) in enumerate(top_3, 1):
                bar_length = int(conf / 5)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {i}. {c_type:<25} {bar} {conf:.1f}%")
        else:
            print("❌ Error processing image")

elif choice == '2':
    # Batch prediction
    folder_path = input("\nEnter folder path: ")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
    else:
        # Find all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            import glob
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        print(f"\n   Found {len(image_files)} images")
        
        results = []
        for img_file in image_files:
            cancer_type, confidence, _ = predict_cancer(img_file)
            if cancer_type:
                results.append({
                    'image': os.path.basename(img_file),
                    'predicted_cancer': cancer_type,
                    'confidence': confidence
                })
                print(f"   ✓ {os.path.basename(img_file)}: {cancer_type} ({confidence:.1f}%)")
        
        # Save results
        if results:
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
    # Test with sample images from your dataset
    print("\n📁 Testing with sample images from dataset...")
    
    dataset_path = r"C:\Ashish\test\dataset\Multi Cancer"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please update the path in the script")
    else:
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        correct = 0
        total = 0
        
        for cancer_type in cancer_types:
            # Clean cancer type name for folder (remove suffix)
            folder_name = cancer_type.replace(' (Leukemia)', '')
            cancer_folder = os.path.join(dataset_path, folder_name)
            
            if os.path.exists(cancer_folder):
                # Find first image in folder
                images = []
                for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                    import glob
                    images.extend(glob.glob(os.path.join(cancer_folder, f'*{ext}')))
                    images.extend(glob.glob(os.path.join(cancer_folder, f'*{ext.upper()}')))
                
                if images:
                    test_image = images[0]
                    pred_type, confidence, _ = predict_cancer(test_image)
                    
                    if pred_type:
                        total += 1
                        is_correct = (pred_type == cancer_type)
                        if is_correct:
                            correct += 1
                            status = "✅"
                        else:
                            status = "❌"
                        
                        print(f"\n{status} {cancer_type}")
                        print(f"   Image: {os.path.basename(test_image)}")
                        print(f"   Predicted: {pred_type}")
                        print(f"   Confidence: {confidence:.1f}%")
        
        if total > 0:
            accuracy = (correct / total) * 100
            print("\n" + "="*60)
            print(f"✅ Test Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
            print("="*60)

print("\n" + "="*60)
print("PREDICTION COMPLETE!")
print("="*60)
