import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Fashion MNIST Model Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the dashboard more attractive
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
        color: #2C3E50;
        font-weight: 600;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #3498DB;
    }
    .metric-label {
        font-size: 16px;
        color: #7F8C8D;
    }
    .section-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #7F8C8D;
    }
    .highlight {
        background-color: #E8F6FC;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Fashion MNIST Model Performance Dashboard")
st.markdown("This dashboard presents comprehensive performance metrics and visualizations for the Fashion MNIST CNN model.")

# Check if the results directory exists
if not os.path.exists('data/results'):
    st.warning("Results directory not found. Please run model_evaluation.py first to generate results.")
    st.stop()

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This dashboard provides a comprehensive view of the CNN model's performance on the Fashion MNIST dataset.

The model was trained to classify 10 different categories of clothing items.

Navigate through different sections to explore various metrics and visualizations.
""")

# Main metrics section
st.header("Model Performance Overview")

# Load model metrics
try:
    with open('data/results/model_metrics.txt', 'r') as f:
        metrics_lines = f.readlines()
    
    test_loss = float(metrics_lines[0].split(': ')[1])
    test_accuracy = float(metrics_lines[1].split(': ')[1])
    
    # Create metrics cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2%}</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """.format(test_accuracy), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.4f}</div>
            <div class="metric-label">Test Loss</div>
        </div>
        """.format(test_loss), unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading model metrics: {e}")

# Load comprehensive results
try:
    results_df = pd.read_csv('data/results/comprehensive_results.csv')
    
    st.subheader("Performance Metrics by Class")
    st.dataframe(results_df.style.format({
        'Accuracy (%)': '{:.2f}%',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}'
    }))
    
    # Highlight best and worst performing classes
    best_class_idx = results_df['Accuracy (%)'].iloc[:-1].idxmax()  # Exclude the average row
    worst_class_idx = results_df['Accuracy (%)'].iloc[:-1].idxmin()  # Exclude the average row
    
    best_class = results_df.iloc[best_class_idx]
    worst_class = results_df.iloc[worst_class_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("#### Best Performing Class")
        st.markdown(f"**{best_class['Class']}** with accuracy of **{best_class['Accuracy (%)']:.2f}%**")
        st.markdown(f"Precision: {best_class['Precision']:.4f}, Recall: {best_class['Recall']:.4f}, F1-Score: {best_class['F1-Score']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("#### Worst Performing Class")
        st.markdown(f"**{worst_class['Class']}** with accuracy of **{worst_class['Accuracy (%)']:.2f}%**")
        st.markdown(f"Precision: {worst_class['Precision']:.4f}, Recall: {worst_class['Recall']:.4f}, F1-Score: {worst_class['F1-Score']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading comprehensive results: {e}")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "Performance Metrics", "ROC & PR Curves", "Examples"])

with tab1:
    st.subheader("Confusion Matrix")
    try:
        confusion_matrix_img = Image.open('data/results/confusion_matrix.png')
        st.image(confusion_matrix_img, use_column_width=True)
        st.markdown("""
        The confusion matrix shows the number of correct and incorrect predictions for each class.
        The diagonal elements represent the number of correct predictions, while off-diagonal elements are incorrect predictions.
        """)
    except Exception as e:
        st.error(f"Error loading confusion matrix: {e}")

with tab2:
    st.subheader("Performance Metrics Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            class_accuracy_img = Image.open('data/results/class_accuracy.png')
            st.image(class_accuracy_img, use_column_width=True)
            st.markdown("Class-wise accuracy shows how well the model performs for each clothing category.")
        except Exception as e:
            st.error(f"Error loading class accuracy chart: {e}")
    
    with col2:
        try:
            precision_recall_f1_img = Image.open('data/results/precision_recall_f1.png')
            st.image(precision_recall_f1_img, use_column_width=True)
            st.markdown("""
            - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
            - **Recall**: The ratio of correctly predicted positive observations to all observations in actual class.
            - **F1-Score**: The weighted average of Precision and Recall.
            """)
        except Exception as e:
            st.error(f"Error loading precision-recall-f1 chart: {e}")

with tab3:
    st.subheader("ROC and Precision-Recall Curves")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            roc_curves_img = Image.open('data/results/roc_curves.png')
            st.image(roc_curves_img, use_column_width=True)
            st.markdown("""
            **ROC (Receiver Operating Characteristic) Curves** show the trade-off between true positive rate and false positive rate.
            The Area Under the Curve (AUC) is a measure of the model's ability to distinguish between classes.
            """)
        except Exception as e:
            st.error(f"Error loading ROC curves: {e}")
    
    with col2:
        try:
            pr_curves_img = Image.open('data/results/precision_recall_curves.png')
            st.image(pr_curves_img, use_column_width=True)
            st.markdown("""
            **Precision-Recall Curves** show the trade-off between precision and recall for different thresholds.
            High precision relates to a low false positive rate, while high recall relates to a low false negative rate.
            """)
        except Exception as e:
            st.error(f"Error loading precision-recall curves: {e}")

with tab4:
    st.subheader("Prediction Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            misclassified_img = Image.open('data/results/misclassified_examples.png')
            st.image(misclassified_img, use_column_width=True)
            st.markdown("**Misclassified Examples** show instances where the model made incorrect predictions.")
        except Exception as e:
            st.error(f"Error loading misclassified examples: {e}")
    
    with col2:
        try:
            confidence_img = Image.open('data/results/confidence_analysis.png')
            st.image(confidence_img, use_column_width=True)
            st.markdown("""
            **Confidence Analysis** shows:
            - Most confident correct predictions
            - Least confident correct predictions
            - Most confident incorrect predictions (dangerous mistakes)
            """)
        except Exception as e:
            st.error(f"Error loading confidence analysis: {e}")

# Model architecture section
st.header("Model Architecture")

try:
    # Try to load model architecture image
    if os.path.exists('data/results/model_architecture.png'):
        model_arch_img = Image.open('data/results/model_architecture.png')
        st.image(model_arch_img, use_column_width=True)
    else:
        # Load model summary from text file
        with open('data/results/model_summary.txt', 'r') as f:
            model_summary = f.read()
        
        st.code(model_summary)
except Exception as e:
    st.error(f"Error loading model architecture: {e}")

# Key findings and insights section
st.header("Key Findings and Insights")

# Create an expander for the analysis
with st.expander("Model Performance Analysis", expanded=True):
    st.markdown("""
    ### Strengths:
    - The model achieves a high overall accuracy on the test set
    - Some categories like Trousers and Ankle boots are classified with very high accuracy
    - The model shows good precision-recall balance across most classes
    
    ### Limitations:
    - The model struggles to differentiate between similar clothing items (e.g., Shirt vs T-shirt)
    - Some categories have more variability and thus lower accuracy
    - There's room for improvement in handling edge cases
    
    ### Improvement Opportunities:
    - Data augmentation to increase training examples for challenging classes
    - Fine-tuning the model architecture with more convolutional layers
    - Implementing attention mechanisms to focus on discriminative features
    """)

# Download section
st.header("Download Results")

# Function to get file as downloadable
def get_binary_file_downloader_html(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'
    return href

# Create a two-column layout for download buttons
col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists('data/results/comprehensive_results.csv'):
        csv = pd.read_csv('data/results/comprehensive_results.csv').to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv,
            file_name="fashion_mnist_results.csv",
            mime="text/csv"
        )

with col2:
    if os.path.exists('data/results/classification_report.txt'):
        with open('data/results/classification_report.txt', 'r') as f:
            classification_report = f.read()
        st.download_button(
            label="Download Classification Report",
            data=classification_report,
            file_name="classification_report.txt",
            mime="text/plain"
        )

with col3:
    if os.path.exists('data/results/model_summary.txt'):
        with open('data/results/model_summary.txt', 'r') as f:
            model_summary = f.read()
        st.download_button(
            label="Download Model Summary",
            data=model_summary,
            file_name="model_summary.txt",
            mime="text/plain"
        )

# Footer
st.markdown("<div class='footer'>Fashion MNIST Classification Model Performance Analysis</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    import base64  # Import here to avoid unused import warning 