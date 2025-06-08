import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
import pandas as pd
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

# Visualize class distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(class_names, train_class_counts)
plt.title('Training Set Class Distribution')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.bar(class_names, test_class_counts)
plt.title('Test Set Class Distribution')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
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
    plt.title(class_names[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig('data/exploration/sample_images.png')
print("Sample images saved to data/exploration/sample_images.png")

# Pixel intensity distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(train_images.flatten(), bins=50)
plt.title('Pixel Intensity Distribution (All Images)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Average image per class
plt.subplot(1, 3, 2)
avg_images = np.zeros((10, 28, 28))
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    avg_images[i] = np.mean(train_images[class_indices], axis=0)

plt.imshow(np.hstack([avg_images[i] for i in range(5)]), cmap='gray')
plt.title('Average Images (Classes 0-4)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.hstack([avg_images[i] for i in range(5, 10)]), cmap='gray')
plt.title('Average Images (Classes 5-9)')
plt.axis('off')
plt.tight_layout()
plt.savefig('data/exploration/pixel_distribution.png')
print("Pixel distribution saved to data/exploration/pixel_distribution.png")

# Image variance analysis
class_variance = np.zeros(10)
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    class_variance[i] = np.mean(np.var(train_images[class_indices], axis=0))

plt.figure(figsize=(10, 5))
plt.bar(class_names, class_variance)
plt.title('Average Pixel Variance by Class')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Variance')
plt.tight_layout()
plt.savefig('data/exploration/class_variance.png')
print("Class variance saved to data/exploration/class_variance.png")

# Correlation matrix between class averages
corr_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        corr_matrix[i, j] = np.corrcoef(avg_images[i].flatten(), 
                                        avg_images[j].flatten())[0, 1]

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Correlation Matrix Between Class Averages')
plt.tight_layout()
plt.savefig('data/exploration/class_correlation.png')
print("Class correlation saved to data/exploration/class_correlation.png")

# t-SNE visualization for dimensionality reduction
from sklearn.manifold import TSNE

# Sample a subset of images for t-SNE (it can be slow on large datasets)
n_samples = 2000
sample_indices = np.random.choice(len(train_images), n_samples, replace=False)
sample_images = train_images[sample_indices].reshape(n_samples, 28*28)
sample_labels = train_labels[sample_indices]

# Apply t-SNE
print("\nPerforming t-SNE dimensionality reduction (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(sample_images)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sample_labels, 
                     cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter, ticks=range(10), label='Class')
plt.title('t-SNE Visualization of Fashion MNIST')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.savefig('data/exploration/tsne_visualization.png')
print("t-SNE visualization saved to data/exploration/tsne_visualization.png")

# Generate a summary report
summary = pd.DataFrame({
    'Class': class_names,
    'Train Count': train_class_counts,
    'Test Count': test_class_counts,
    'Average Variance': class_variance
})
summary.to_csv('data/exploration/class_summary.csv', index=False)
print("Class summary saved to data/exploration/class_summary.csv")

print("\nData exploration analysis completed!") 