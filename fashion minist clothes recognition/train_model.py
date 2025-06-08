import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load and preprocess the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images to [0, 1] range
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape the images to 28x28x1 for CNN
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# One-hot encode the labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Create class names array for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Save class names to file for the Streamlit app
np.save('data/class_names.npy', class_names)

# Create the CNN model
print("Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Create a checkpoint callback
checkpoint = ModelCheckpoint('model/fashion_mnist_model.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

# Train the model
print("Training model...")
history = model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint]
)

# Evaluate the model
print("Evaluating model...")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Save the final model
model.save('model/fashion_mnist_model.h5')
print("Model saved to model/fashion_mnist_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('data/training_history.png')
print("Training history saved to data/training_history.png")

# Save some test images for the demo
print("Saving sample test images...")
os.makedirs('images', exist_ok=True)
np.random.seed(42)
for i in range(10):
    idx = np.random.randint(0, len(test_images))
    img = test_images[idx] * 255
    img = img.reshape(28, 28).astype(np.uint8)
    plt.imsave(f'images/sample_{i}_{class_names[np.argmax(test_labels[idx])]}.png', img, cmap='gray')

print("Sample images saved in images/ directory")
print("All done! You can now run the Streamlit app with 'streamlit run app.py'") 