import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import os

# Create directories if they don't exist
os.makedirs('data/exploration', exist_ok=True)

# Load the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

# Save class distribution to file
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(10), train_class_counts)
plt.title('Training Set Class Distribution')
plt.xlabel('Class Index')
plt.ylabel('Count')
plt.xticks(range(10))

plt.subplot(1, 2, 2)
plt.bar(range(10), test_class_counts)
plt.title('Test Set Class Distribution')
plt.xlabel('Class Index')
plt.ylabel('Count')
plt.xticks(range(10))
plt.tight_layout()
plt.savefig('data/exploration/class_distribution.png')
print("Class distribution saved to data/exploration/class_distribution.png")

# Image visualization - sample from each class
plt.figure(figsize=(12, 10))
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    random_idx = np.random.choice(class_indices)
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[random_idx], cmap='gray')
    plt.title(f'Class {i}: {class_names[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('data/exploration/sample_images.png')
print("Sample images saved to data/exploration/sample_images.png")

# Generate summary statistics
mean_per_class = []
std_per_class = []
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    class_pixels = train_images[class_indices].mean(axis=0).flatten()
    mean_per_class.append(np.mean(class_pixels))
    std_per_class.append(np.std(class_pixels))

print("\n--- Pixel Statistics by Class ---")
for i, name in enumerate(class_names):
    print(f"{name}: Mean={mean_per_class[i]:.2f}, Std={std_per_class[i]:.2f}")

# Save pixel intensity histogram
plt.figure(figsize=(10, 6))
plt.hist(train_images.flatten(), bins=50)
plt.title('Pixel Intensity Distribution (All Training Images)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('data/exploration/pixel_histogram.png')
print("Pixel histogram saved to data/exploration/pixel_histogram.png")

print("\nData exploration analysis completed!") 