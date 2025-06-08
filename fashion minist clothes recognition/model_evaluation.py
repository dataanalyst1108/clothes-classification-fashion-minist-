import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os
import time

# Create directories if they don't exist
os.makedirs('data/results', exist_ok=True)

# Load class names
def load_class_names():
    try:
        class_names = np.load('data/class_names.npy', allow_pickle=True)
        return class_names
    except Exception as e:
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def evaluate_model():
    print("Loading Fashion MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Normalize and reshape images
    test_images = test_images.astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1)
    
    # One-hot encode the labels for model evaluation
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, 10)
    
    # Load the trained model
    try:
        print("Loading model...")
        model = load_model('model/fashion_mnist_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run train_model.py first to generate the model.")
        return
    
    # Load class names
    class_names = load_class_names()
    
    # Model evaluation
    print("\n--- Model Evaluation ---\n")
    
    start_time = time.time()
    
    # 1. Overall model evaluation
    print("1. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save these metrics to a txt file
    with open('data/results/model_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    # 2. Make predictions
    print("2. Making predictions...")
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 3. Confusion Matrix
    print("3. Generating confusion matrix...")
    conf_matrix = confusion_matrix(test_labels, predicted_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('data/results/confusion_matrix.png')
    print("Confusion matrix saved to data/results/confusion_matrix.png")
    
    # 4. Classification Report
    print("4. Generating classification report...")
    class_report = classification_report(test_labels, predicted_classes, 
                                         target_names=class_names, 
                                         output_dict=True)
    
    # Save the report to a txt file
    with open('data/results/classification_report.txt', 'w') as f:
        f.write(classification_report(test_labels, predicted_classes, target_names=class_names))
    
    # Convert to DataFrame for easier manipulation
    report_df = pd.DataFrame(class_report).transpose()
    
    # 5. Precision, Recall, F1-Score per class
    print("5. Generating precision, recall, and F1-score charts...")
    metrics_df = report_df.iloc[:-3].copy()  # Exclude the avg rows
    
    # Plot precision, recall and f1-score
    plt.figure(figsize=(14, 8))
    bar_width = 0.25
    index = np.arange(len(class_names))
    
    plt.bar(index, metrics_df['precision'], bar_width, label='Precision', color='#3498db')
    plt.bar(index + bar_width, metrics_df['recall'], bar_width, label='Recall', color='#2ecc71')
    plt.bar(index + 2*bar_width, metrics_df['f1-score'], bar_width, label='F1-Score', color='#e74c3c')
    
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Precision, Recall, F1-Score by Class', fontsize=16)
    plt.xticks(index + bar_width, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/results/precision_recall_f1.png')
    print("Precision, recall, and F1-score chart saved to data/results/precision_recall_f1.png")
    
    # 6. Class-wise accuracy
    print("6. Generating class-wise accuracy chart...")
    class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracy * 100)
    plt.title('Class-wise Accuracy', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('data/results/class_accuracy.png')
    print("Class-wise accuracy chart saved to data/results/class_accuracy.png")
    
    # 7. Misclassified examples
    print("7. Generating misclassified examples visualization...")
    misclassified_indices = np.where(predicted_classes != test_labels)[0]
    
    # Choose a random subset of misclassified examples to display
    np.random.seed(42)
    if len(misclassified_indices) > 15:
        display_indices = np.random.choice(misclassified_indices, 15, replace=False)
    else:
        display_indices = misclassified_indices
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(display_indices):
        plt.subplot(3, 5, i+1)
        img = test_images[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {class_names[test_labels[idx]]}\nPred: {class_names[predicted_classes[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/results/misclassified_examples.png')
    print("Misclassified examples saved to data/results/misclassified_examples.png")
    
    # 8. Top confident and least confident predictions
    print("8. Generating confidence analysis...")
    # Get the max probability (confidence) for each prediction
    confidence = np.max(predictions, axis=1)
    
    # Most confident correct predictions
    correct_indices = np.where(predicted_classes == test_labels)[0]
    correct_confidences = confidence[correct_indices]
    most_confident_correct = correct_indices[np.argsort(correct_confidences)[-5:]]  # Top 5
    
    # Least confident correct predictions
    least_confident_correct = correct_indices[np.argsort(correct_confidences)[:5]]  # Bottom 5
    
    # Most confident incorrect predictions (most dangerous mistakes)
    incorrect_indices = misclassified_indices
    if len(incorrect_indices) > 0:
        incorrect_confidences = confidence[incorrect_indices]
        most_confident_incorrect = incorrect_indices[np.argsort(incorrect_confidences)[-5:]]  # Top 5
    else:
        most_confident_incorrect = []
    
    # Plot
    plt.figure(figsize=(15, 12))
    
    # Most confident correct
    for i, idx in enumerate(most_confident_correct):
        plt.subplot(3, 5, i+1)
        img = test_images[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"{class_names[test_labels[idx]]}\nConf: {confidence[idx]:.1%}")
        plt.axis('off')
    plt.subplot(3, 5, 3)
    plt.text(0.5, 0.5, 'Most Confident\nCorrect Predictions', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Least confident correct
    for i, idx in enumerate(least_confident_correct):
        plt.subplot(3, 5, i+6)
        img = test_images[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"{class_names[test_labels[idx]]}\nConf: {confidence[idx]:.1%}")
        plt.axis('off')
    plt.subplot(3, 5, 8)
    plt.text(0.5, 0.5, 'Least Confident\nCorrect Predictions', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Most confident incorrect
    if len(most_confident_incorrect) > 0:
        for i, idx in enumerate(most_confident_incorrect[:4]):
            plt.subplot(3, 5, i+11)
            img = test_images[idx].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(f"True: {class_names[test_labels[idx]]}\nPred: {class_names[predicted_classes[idx]]}\nConf: {confidence[idx]:.1%}")
            plt.axis('off')
    plt.subplot(3, 5, 13)
    plt.text(0.5, 0.5, 'Most Confident\nIncorrect Predictions', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/results/confidence_analysis.png')
    print("Confidence analysis saved to data/results/confidence_analysis.png")
    
    # 9. ROC curve and AUC for each class
    print("9. Generating ROC curves...")
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(10):
        # Convert labels to one-hot encoding for ROC computation
        true_labels = (test_labels == i).astype(int)
        predicted_probs = predictions[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(true_labels, predicted_probs)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                    'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('data/results/roc_curves.png')
    print("ROC curves saved to data/results/roc_curves.png")
    
    # 10. Precision-Recall curves
    print("10. Generating Precision-Recall curves...")
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Compute Precision-Recall curve and average precision for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(10):
        true_labels = (test_labels == i).astype(int)
        predicted_probs = predictions[:, i]
        
        precision[i], recall[i], _ = precision_recall_curve(true_labels, predicted_probs)
        avg_precision[i] = average_precision_score(true_labels, predicted_probs)
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                    'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(10), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves', fontsize=16)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('data/results/precision_recall_curves.png')
    print("Precision-Recall curves saved to data/results/precision_recall_curves.png")
    
    # 11. Summarize model architecture
    print("11. Generating model architecture summary...")
    from tensorflow.keras.utils import plot_model
    
    # Save model summary to a text file
    with open('data/results/model_summary.txt', 'w') as f:
        # Redirect stdout to the file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        model.summary()
        sys.stdout = original_stdout
    
    # Save model architecture as an image if pydot is available
    try:
        plot_model(model, to_file='data/results/model_architecture.png', show_shapes=True, show_layer_names=True)
        print("Model architecture diagram saved to data/results/model_architecture.png")
    except Exception as e:
        print(f"Could not generate model architecture diagram: {e}")
        print("You may need to install pydot and graphviz for this feature.")
    
    # 12. Final comprehensive results table
    print("12. Generating comprehensive results table...")
    
    # Prepare data
    results_data = {
        'Class': class_names,
        'Accuracy (%)': class_accuracy * 100,
        'Precision': metrics_df['precision'].values,
        'Recall': metrics_df['recall'].values,
        'F1-Score': metrics_df['f1-score'].values,
        'Support': metrics_df['support'].values
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Add means
    results_df.loc['avg/total'] = ['Average'] + [results_df[col].mean() for col in results_df.columns[1:-1]] + [results_df['Support'].sum()]
    
    # Save to CSV
    results_df.to_csv('data/results/comprehensive_results.csv', index=False)
    print("Comprehensive results saved to data/results/comprehensive_results.csv")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nModel evaluation completed in {duration:.2f} seconds.")
    print(f"All results and visualizations saved to data/results/ directory.")
    
    return test_accuracy

if __name__ == "__main__":
    evaluate_model() 