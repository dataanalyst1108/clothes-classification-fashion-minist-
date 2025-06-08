# Fashion MNIST Clothes Recognition

A CNN-based Fashion MNIST clothes recognition system with Streamlit deployment. The model can classify 10 different categories of clothing items with high accuracy.

## Features

- **CNN Model**: A convolutional neural network trained on the Fashion MNIST dataset
- **Exploratory Data Analysis**: Comprehensive data visualization and analysis of the Fashion MNIST dataset
- **Model Evaluation**: Detailed performance metrics, visualizations, and analysis tools
- **Performance Dashboard**: Interactive Streamlit dashboard for exploring model results
- **Streamlit Web App**: An attractive and user-friendly interface for image classification
- **Custom Image Testing**: Upload your own images or use sample images to test the model
- **Visual Results**: See confidence scores and processed images

## Categories

The model can recognize the following 10 categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis (Optional)

Run the EDA script to explore the Fashion MNIST dataset:

```
python fashion_mnist_eda.py
```

This will generate detailed visualizations and analyses in the `data/eda/` directory, including:
- Class distribution analysis
- Sample images from each class
- Pixel intensity distribution
- Image variance analysis
- Class similarity and confusion analysis
- Average images per class

### 2. Train the Model

Run the training script to download the Fashion MNIST dataset, train the CNN model, and generate sample images:

```
python train_model.py
```

This will:
- Download and preprocess the Fashion MNIST dataset
- Train a CNN model with multiple convolutional and dense layers
- Save the trained model to `model/fashion_mnist_model.h5`
- Generate sample test images in the `images/` directory
- Save training history plots

### 3. Evaluate Model Performance

Run the model evaluation script to generate comprehensive performance metrics and visualizations:

```
python model_evaluation.py
```

This will generate various metrics and visualizations in the `data/results/` directory, including:
- Overall model metrics (accuracy, loss)
- Confusion matrix
- Class-wise accuracy, precision, recall, and F1-score
- ROC curves and AUC for each class
- Precision-Recall curves
- Examples of correctly and incorrectly classified images
- Comprehensive results table

### 4. View Performance Dashboard

Launch the model performance dashboard to explore the results through an interactive interface:

```
streamlit run model_results_dashboard.py
```

This will open a web browser with an interactive dashboard featuring:
- Key performance metrics
- Interactive visualizations
- Detailed analysis of model strengths and weaknesses
- Ability to download results

### 5. Test Custom Images (Optional)

If you want to test your own images outside the web app:

```
python test_custom_images.py
```

Place your images in the `images/` directory first, and the script will generate prediction results in `images/test_results/`.

### 6. Launch the Streamlit App

Start the web application for image classification:

```
streamlit run app.py
```

This will launch a web browser with the application where you can:
- Upload your own fashion item images
- Try sample images generated from the test dataset
- View predictions with confidence scores
- See how the image is processed for classification

## Project Structure

```
fashion-mnist-recognition/
├── model/                  # Directory for saved models
├── data/                   # Directory for class names and training history
│   ├── eda/                # Data exploration visualizations
│   └── results/            # Model evaluation results and metrics
├── images/                 # Sample images and test results
├── app.py                  # Streamlit classification web application
├── train_model.py          # Script to train the CNN model
├── fashion_mnist_eda.py    # Exploratory data analysis script
├── model_evaluation.py     # Model performance evaluation script
├── model_results_dashboard.py # Performance metrics dashboard
├── test_custom_images.py   # Script to test custom images
└── requirements.txt        # Required Python packages
```

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Dropout for regularization
- Dense layers for classification

## Model Performance

The model achieves high accuracy on the Fashion MNIST test set. Key metrics include:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix showing classification performance

For detailed performance metrics and visualizations, run the evaluation script and view the results dashboard.

## Technical Details

- Framework: TensorFlow/Keras
- Frontend: Streamlit
- Image Processing: Pillow, NumPy
- Visualization: Matplotlib, Seaborn
- Analytics: Scikit-learn

## Customization

You can modify the model architecture in `train_model.py` to experiment with different configurations. Adjust the number of epochs, batch size, or layer parameters to potentially improve performance. 