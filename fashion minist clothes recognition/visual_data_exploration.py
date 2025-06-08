import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
import random

# Create directories if they don't exist
os.makedirs('data/visualizations', exist_ok=True)

# Load the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images (for visualization purposes)
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# Define class names 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Figure 1: Sample images from each class
plt.figure(figsize=(15, 8))
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    for j in range(5):  # 5 samples per class
        plt.subplot(10, 5, i*5 + j + 1)
        sample_idx = random.choice(class_indices)
        plt.imshow(train_images[sample_idx], cmap='gray')
        if j == 0:
            plt.ylabel(class_names[i], rotation=45, fontsize=10)
        plt.xticks([])
        plt.yticks([])
plt.tight_layout()
plt.savefig('data/visualizations/sample_images_by_class.png')
print("Sample images visualization saved")

# Figure 2: Average image per class
plt.figure(figsize=(15, 4))
avg_images = np.zeros((10, 28, 28))
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    avg_images[i] = np.mean(train_images[class_indices], axis=0)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(avg_images[i], cmap='viridis')
    plt.title(class_names[i])
    plt.axis('off')
plt.suptitle('Average Image per Class')
plt.tight_layout()
plt.savefig('data/visualizations/average_images_by_class.png')
print("Average images visualization saved")

# Figure 3: Pixel intensity distribution
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    class_indices = np.where(train_labels == i)[0]
    plt.hist(train_images[class_indices].flatten(), bins=50, alpha=0.7)
    plt.title(class_names[i])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('data/visualizations/pixel_distributions_1.png')

plt.figure(figsize=(15, 5))
for i in range(5, 10):
    plt.subplot(1, 5, i-4)
    class_indices = np.where(train_labels == i)[0]
    plt.hist(train_images[class_indices].flatten(), bins=50, alpha=0.7)
    plt.title(class_names[i])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('data/visualizations/pixel_distributions_2.png')
print("Pixel distribution visualizations saved")

# Figure 4: Pixel intensity variance
plt.figure(figsize=(12, 8))
variance_maps = np.zeros((10, 28, 28))
for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    variance_maps[i] = np.var(train_images_normalized[class_indices], axis=0)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(variance_maps[i], cmap='hot')
    plt.title(class_names[i])
    plt.axis('off')
plt.suptitle('Pixel Variance Maps by Class')
plt.tight_layout()
plt.savefig('data/visualizations/variance_maps.png')
print("Variance maps visualization saved")

# Figure 5: t-SNE visualization
print("Computing t-SNE (this may take a while)...")
# Sample a subset for t-SNE to speed up computation
n_samples = 2000
indices = np.random.choice(train_images.shape[0], n_samples, replace=False)
sample_images = train_images[indices].reshape(n_samples, -1) / 255.0
sample_labels = train_labels[indices]

# Compute t-SNE
try:
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(sample_images)
    
    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    for i in range(10):
        class_indices = np.where(sample_labels == i)[0]
        plt.scatter(tsne_result[class_indices, 0], tsne_result[class_indices, 1], 
                   label=class_names[i], alpha=0.6, s=5)
    plt.legend(fontsize=10)
    plt.title('t-SNE visualization of Fashion MNIST')
    plt.savefig('data/visualizations/tsne_visualization.png')
    print("t-SNE visualization saved")
except Exception as e:
    print(f"Error computing t-SNE: {e}")

# Figure 6: Correlation matrix between classes
plt.figure(figsize=(12, 10))
correlation_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        correlation_matrix[i, j] = np.corrcoef(
            avg_images[i].flatten(), 
            avg_images[j].flatten()
        )[0, 1]

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Correlation Between Class Averages')
plt.tight_layout()
plt.savefig('data/visualizations/class_correlation_matrix.png')
print("Correlation matrix visualization saved")

# Figure 7: Class distribution
plt.figure(figsize=(12, 6))
train_counts = np.bincount(train_labels)
test_counts = np.bincount(test_labels)

x = np.arange(len(class_names))
width = 0.35

plt.bar(x - width/2, train_counts, width, label='Training Set')
plt.bar(x + width/2, test_counts, width, label='Test Set')
plt.xticks(x, class_names, rotation=45, ha='right')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('data/visualizations/class_distribution.png')
print("Class distribution visualization saved")

print("\nAll visualizations have been saved to data/visualizations/") 