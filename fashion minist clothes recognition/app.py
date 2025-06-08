import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more attractive
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTitle {
        color: #2C3E50;
        font-weight: 800;
    }
    .stHeader {
        font-weight: 600;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #3498DB;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        color: white;
    }
    .stSidebar {
        background-color: #2C3E50;
        color: white;
    }
    .prediction-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        background-color: #E0E0E0;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .confidence-value {
        height: 100%;
        border-radius: 10px;
        background-color: #27AE60;
        color: white;
        text-align: center;
        line-height: 20px;
        font-weight: bold;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('model/fashion_mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load class names
@st.cache_data
def load_class_names():
    try:
        class_names = np.load('data/class_names.npy', allow_pickle=True)
        return class_names
    except Exception as e:
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the image
def preprocess_image(image, target_size=(28, 28)):
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# Function to make predictions
def predict(image, model, class_names):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Get the top 3 predictions
    top_3_indices = predictions[0].argsort()[-3:][::-1]
    top_3_classes = [class_names[i] for i in top_3_indices]
    top_3_probabilities = [predictions[0][i] * 100 for i in top_3_indices]
    
    return top_3_classes, top_3_probabilities, processed_image.reshape(28, 28)

# Main function
def main():
    st.title("👕 Fashion Item Classifier")
    
    # Load model and class names
    model = load_trained_model()
    class_names = load_class_names()
    
    if model is None:
        st.warning("The model couldn't be loaded. Please train the model first by running train_model.py")
        return
    
    # Create sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This application uses a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset to classify clothing items.
    
    The model can recognize 10 different categories of clothing items.
    """)
    
    st.sidebar.header("Classes")
    for i, class_name in enumerate(class_names):
        st.sidebar.write(f"{i}: {class_name}")
    
    # Show model performance if available
    if os.path.exists('data/training_history.png'):
        st.sidebar.header("Model Performance")
        st.sidebar.image('data/training_history.png', caption="Training History", use_column_width=True)
    
    # Main content
    st.header("Upload an Image")
    st.write("Upload an image of a clothing item to classify it.")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Or use sample images
        st.markdown("### Or try a sample image:")
        
        # Get sample images from the images directory
        sample_images = []
        if os.path.exists('images'):
            sample_images = [f for f in os.listdir('images') if f.endswith(('.png', '.jpg', '.jpeg')) and 'test_results' not in f]
        
        if sample_images:
            # Create a grid of buttons for sample images
            n_cols = 3
            sample_rows = [sample_images[i:i+n_cols] for i in range(0, len(sample_images), n_cols)]
            
            for row in sample_rows:
                cols = st.columns(n_cols)
                for i, img_name in enumerate(row):
                    with cols[i]:
                        if st.button(f"{img_name.split('_')[-1].split('.')[0]}"):
                            with open(os.path.join('images', img_name), "rb") as f:
                                uploaded_file = io.BytesIO(f.read())
        else:
            st.info("No sample images found. Run train_model.py to generate sample images.")
    
    # If an image is uploaded
    if uploaded_file is not None:
        with col2:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction when user clicks the button
            if st.button("Classify Image"):
                with st.spinner("Analyzing..."):
                    # Add a small delay for effect
                    time.sleep(1)
                    
                    # Get predictions
                    top_classes, top_probs, processed_img = predict(image, model, class_names)
                    
                    # Display predictions
                    st.subheader("Prediction Results")
                    
                    # Display top prediction
                    st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; color: #2C3E50;'>This is a {top_classes[0]}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='confidence-meter'><div class='confidence-value' style='width: {top_probs[0]}%;'>{top_probs[0]:.1f}%</div></div>", unsafe_allow_html=True)
                    st.markdown(f"</div>", unsafe_allow_html=True)
                    
                    # Display all top 3 predictions
                    for i in range(len(top_classes)):
                        st.text(f"{i+1}. {top_classes[i]}: {top_probs[i]:.1f}%")
                    
                    # Display the processed image
                    st.subheader("Processed Image (28x28)")
                    st.image(processed_img, caption="Processed for Classification", width=150)
    
    # Footer
    st.markdown("<div class='footer'>Developed with ❤️ for Fashion MNIST Classification</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 