import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
import os

# Create directories if they don't exist
os.makedirs('data/eda', exist_ok=True)

# Load the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Dataset loaded successfully.")
print(f"Training set shape: {train_images.shape}")
print(f"Test set shape: {test_images.shape}")

# Data exploration and visualizations
print("\n--- Data Exploration and Analysis ---\n")

# 1. Class distribution in training set
print("1. Class Distribution in Training Set")
unique_classes, class_counts = np.unique(train_labels, return_counts=True)
class_distribution = pd.DataFrame({
    'Class': [class_names[i] for i in unique_classes],
    'Count': class_counts,
    'Percentage': class_counts / len(train_labels) * 100
})
print(class_distribution.to_string(index=False))

# Visualize class distribution
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, class_counts)
plt.title('Class Distribution in Training Set', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add counts on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{height}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data/eda/class_distribution.png')
print("Class distribution chart saved to data/eda/class_distribution.png")

# 2. Sample images from each class
plt.figure(figsize=(12, 10))
for i in range(10):
    # Find all indices for current class
    indices = np.where(train_labels == i)[0]
    
    # Randomly select 5 images
    np.random.seed(42+i)  # for reproducibility
    selected_indices = np.random.choice(indices, 5, replace=False)
    
    for j, idx in enumerate(selected_indices):
        plt.subplot(10, 5, i*5 + j + 1)
        plt.imshow(train_images[idx], cmap='gray')
        plt.title(f"{class_names[i]}")
        plt.axis('off')

plt.tight_layout()
plt.savefig('data/eda/sample_images.png')
print("Sample images saved to data/eda/sample_images.png")

# 3. Pixel intensity distribution
plt.figure(figsize=(12, 6))

# Overall pixel intensity
plt.subplot(1, 2, 1)
plt.hist(train_images.ravel(), bins=50, color='gray', alpha=0.7)
plt.title('Overall Pixel Intensity Distribution', fontsize=14)
plt.xlabel('Pixel Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Average pixel intensity per class
plt.subplot(1, 2, 2)
avg_intensity = [train_images[train_labels == i].mean() for i in range(10)]
bars = plt.bar(class_names, avg_intensity, color='orange')
plt.title('Average Pixel Intensity per Class', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Average Intensity', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data/eda/pixel_intensity.png')
print("Pixel intensity analysis saved to data/eda/pixel_intensity.png")

# 4. Image variance analysis
plt.figure(figsize=(12, 6))

# Calculate variance for each image
image_variances = np.var(train_images.reshape(train_images.shape[0], -1), axis=1)

# Average variance per class
avg_var_per_class = [np.mean(image_variances[train_labels == i]) for i in range(10)]

bars = plt.bar(class_names, avg_var_per_class)
plt.title('Average Image Variance per Class', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Average Variance', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data/eda/image_variance.png')
print("Image variance analysis saved to data/eda/image_variance.png")

# 5. Correlation between classes
print("\n5. Class Similarity Analysis (Most Easily Confused Classes)")

# Create a confusion matrix to see which classes might be easily confused
confusion_matrix = np.zeros((10, 10))

# For each image, calculate distance to average image of each class
class_avg_images = [np.mean(train_images[train_labels == i], axis=0) for i in range(10)]

# Calculate distances for a subset of images to save time
np.random.seed(42)
sample_indices = np.random.choice(len(train_images), 1000, replace=False)

for idx in sample_indices:
    img = train_images[idx]
    true_label = train_labels[idx]
    
    # Calculate MSE between this image and average images of each class
    mse_distances = [np.mean((img - avg_img) ** 2) for avg_img in class_avg_images]
    
    # Sort distances (excluding the true class)
    sorted_distances = np.argsort(mse_distances)
    
    # If the second closest class is not the true class
    # Note: sorted_distances[0] should be the closest class
    if sorted_distances[0] != true_label:
        # Increment confusion count
        confusion_matrix[true_label, sorted_distances[0]] += 1
    elif len(sorted_distances) > 1:
        # Increment count for second closest class
        confusion_matrix[true_label, sorted_distances[1]] += 1

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix, annot=True, fmt='.0f', xticklabels=class_names, yticklabels=class_names)
plt.title('Class Confusion Analysis (Based on Image Similarity)', fontsize=16)
plt.xlabel('Confused With', fontsize=14)
plt.ylabel('True Class', fontsize=14)
plt.tight_layout()
plt.savefig('data/eda/class_confusion.png')
print("Class confusion analysis saved to data/eda/class_confusion.png")

# Print most confused classes
for i in range(10):
    most_confused_idx = np.argsort(confusion_matrix[i])[-2]  # Second highest (highest might be the diagonal)
    if confusion_matrix[i, most_confused_idx] > 0:
        print(f"{class_names[i]} is most easily confused with {class_names[most_confused_idx]} " 
              f"({confusion_matrix[i, most_confused_idx]:.0f} instances)")

# 6. Average image per class
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    # Calculate average image for this class
    avg_img = np.mean(train_images[train_labels == i], axis=0)
    plt.imshow(avg_img, cmap='gray')
    plt.title(f"Avg. {class_names[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('data/eda/average_images.png')
print("Average images per class saved to data/eda/average_images.png")

print("\nExploratory Data Analysis completed. Results saved to data/eda/ directory.") 