import tensorflow as tf

print("="*60)
print("GPU CHECK")
print("="*60)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n✅ GPU(s) found: {len(gpus)}")
    for gpu in gpus:
        print(f"   {gpu}")
    print("\n🚀 GPU acceleration is available!")
    print("   Your training will be much faster")
else:
    print("\n❌ No GPU found")
    print("   Training will be on CPU (slower)")
    print("\n💡 To speed up training:")
    print("   1. Use the 'fast_image_model.py' script")
    print("   2. Reduce image size further")
    print("   3. Use fewer images for initial testing")
    print("   4. Consider using Google Colab (free GPU)")
