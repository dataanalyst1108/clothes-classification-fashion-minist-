import numpy as np
import os
from tensorflow.keras.datasets import fashion_mnist

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)

print("Starting Fashion MNIST data exploration...")

# Load the Fashion MNIST dataset
try:
    print("Loading Fashion MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Define class names 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Basic dataset information
print("\n--- Dataset Information ---")
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Image data type: {train_images.dtype}")
print(f"Pixel value range: [{np.min(train_images)}, {np.max(train_images)}]")

# Class distribution analysis
train_class_counts = np.bincount(train_labels)
test_class_counts = np.bincount(test_labels)

print("\n--- Class Distribution ---")
for i, name in enumerate(class_names):
    print(f"{name}: {train_class_counts[i]} (train), {test_class_counts[i]} (test)")

# Generate summary statistics
print("\n--- Pixel Statistics by Class ---")
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    class_pixels = train_images[class_indices]
    mean_val = np.mean(class_pixels)
    std_val = np.std(class_pixels)
    min_val = np.min(class_pixels)
    max_val = np.max(class_pixels)
    print(f"Class {i} ({class_names[i]}): Mean={mean_val:.2f}, Std={std_val:.2f}, Min={min_val}, Max={max_val}")

# Calculate correlation between class averages
print("\n--- Class Similarity Analysis ---")
class_avgs = []
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    avg_image = np.mean(train_images[class_indices], axis=0).flatten()
    class_avgs.append(avg_image)

print("Most similar class pairs (based on average image correlation):")
similarities = []
for i in range(10):
    for j in range(i+1, 10):
        corr = np.corrcoef(class_avgs[i], class_avgs[j])[0, 1]
        similarities.append((i, j, corr))

# Sort by correlation (highest first)
similarities.sort(key=lambda x: x[2], reverse=True)
for i, j, corr in similarities[:5]:
    print(f"{class_names[i]} and {class_names[j]}: {corr:.4f}")

print("\n--- Class Balance Analysis ---")
total_train = len(train_labels)
total_test = len(test_labels)
for i, name in enumerate(class_names):
    train_pct = (train_class_counts[i] / total_train) * 100
    test_pct = (test_class_counts[i] / total_test) * 100
    print(f"{name}: {train_pct:.1f}% (train), {test_pct:.1f}% (test)")

# Save summary statistics to text file
with open('data/fashion_mnist_summary.txt', 'w') as f:
    f.write("Fashion MNIST Dataset Summary\n")
    f.write("===========================\n\n")
    
    f.write("Dataset Information\n")
    f.write(f"Training images: {train_images.shape}\n")
    f.write(f"Test images: {test_images.shape}\n")
    f.write(f"Pixel value range: [{np.min(train_images)}, {np.max(train_images)}]\n\n")
    
    f.write("Class Distribution\n")
    for i, name in enumerate(class_names):
        f.write(f"{name}: {train_class_counts[i]} (train), {test_class_counts[i]} (test)\n")
    
    f.write("\nPixel Statistics by Class\n")
    for i in range(10):
        class_indices = np.where(train_labels == i)[0]
        class_pixels = train_images[class_indices]
        mean_val = np.mean(class_pixels)
        std_val = np.std(class_pixels)
        f.write(f"Class {i} ({class_names[i]}): Mean={mean_val:.2f}, Std={std_val:.2f}\n")

print(f"\nSummary statistics saved to data/fashion_mnist_summary.txt")
print("\nData exploration analysis completed!") 