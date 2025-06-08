import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os

def load_and_prepare_image(image_path, target_size=(28, 28)):
    """Load and prepare an image for prediction."""
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def main():
    # Load model
    try:
        model = load_model('model/fashion_mnist_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run train_model.py first to generate the model.")
        return
    
    # Load class names
    try:
        class_names = np.load('data/class_names.npy', allow_pickle=True)
        print("Class names loaded successfully.")
    except Exception as e:
        print(f"Error loading class names: {e}")
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        print("Using default class names.")
    
    # Create output directory for test results
    os.makedirs('images/test_results', exist_ok=True)
    
    # Get all image files from the images directory
    image_files = [f for f in os.listdir('images') if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the 'images' directory.")
        return
    
    # Process each image
    for img_file in image_files:
        image_path = os.path.join('images', img_file)
        
        # Skip if it's in test_results directory
        if 'test_results' in image_path:
            continue
            
        print(f"Processing {img_file}...")
        
        # Load and prepare the image
        img_array = load_and_prepare_image(image_path)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        # Get the predicted class name
        predicted_label = class_names[predicted_class]
        
        # Load the original image for display
        img = Image.open(image_path)
        
        # Create a figure to display the image and prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        
        # Save the figure
        output_path = os.path.join('images/test_results', f"result_{img_file}")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Prediction saved to {output_path}")
    
    print("All images processed successfully!")

if __name__ == "__main__":
    main() 