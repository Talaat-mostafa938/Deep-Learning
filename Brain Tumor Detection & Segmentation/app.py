import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2 as cv
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure Streamlit page
st.set_page_config(page_title="Brain Tumor", layout="wide")

st.header('Brain Tumor Detection & Segmentation')

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        detection_model = load_model('brain_tumor_classifier.keras')
        segmentation_model = load_model('brain_tumor_segmentation_model.keras')
        return detection_model, segmentation_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

Brain_Detection, Brain_Segmentation = load_models()

Brain_Detection_classes = ['No', 'Yes']

# Check if models are loaded successfully
if Brain_Detection is None or Brain_Segmentation is None:
    st.error("Models could not be loaded. Please check if the model files exist.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif"])

# Create columns for buttons
col1, col2 = st.columns(2)

with col1:
    detection_button = st.button('Brain Detection', use_container_width=True)

with col2:
    segmentation_button = st.button('Brain Segmentation', use_container_width=True)

if detection_button and uploaded_file is not None:
    try:
        # Load and preprocess image
        img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_normalized = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        with st.spinner('Analyzing image...'):
            predictions = Brain_Detection.predict(img_normalized, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Display results
        st.subheader("Detection Results:")
        if Brain_Detection_classes[predicted_class] == 'No':
            st.success(f'The patient does not have a brain tumor! (Confidence: {confidence:.2f}%)')
        else:
            st.error(f'The patient has a brain tumor! (Confidence: {confidence:.2f}%)')

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Probability bar plot
        sns.barplot(x=Brain_Detection_classes, y=predictions[0], palette='Set3', ax=ax1)
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Probability')
        ax1.set_title('Prediction Probabilities')
        
        # Input image
        ax2.imshow(img_array.astype('uint8'))
        ax2.set_title('Input Image')
        ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")

if segmentation_button and uploaded_file is not None:
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Read and decode image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        
        if img is None:
            st.error("Could not decode the uploaded image. Please try a different image.")
        else:
            # Convert color space and resize
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_resized = cv.resize(img_rgb, (256, 256))
            
            # Normalize and prepare for prediction
            img_input = img_resized / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            # Make prediction
            with st.spinner('Segmenting tumor...'):
                pred = Brain_Segmentation.predict(img_input, verbose=0)[0]
            
            # Create binary mask
            pred_mask = (pred > 0.5).astype(np.uint8)

            # Display results
            st.subheader("Segmentation Results:")
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(img_resized)
            ax1.set_title("Input Image")
            ax1.axis('off')
            
            # Predicted mask
            ax2.imshow(pred_mask.squeeze(), cmap='gray')
            ax2.set_title("Predicted Tumor Mask")
            ax2.axis('off')
            
            # Overlay
            overlay = img_resized.copy()
            if pred_mask.max() > 0:  # If tumor detected
                # Create colored mask overlay
                mask_colored = np.zeros_like(img_resized)
                mask_colored[:, :, 0] = pred_mask.squeeze() * 255  # Red channel
                overlay = cv.addWeighted(img_resized, 0.7, mask_colored, 0.3, 0)
                st.success("Tumor region detected and highlighted!")
            else:
                st.info("No tumor region detected in the segmentation.")
            
            ax3.imshow(overlay)
            ax3.set_title("Tumor Overlay")
            ax3.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"Error during segmentation: {str(e)}")

# Add information section
st.sidebar.header("About")
st.sidebar.info("""
This application uses deep learning models to:
1. **Detect** the presence of brain tumors in MRI images
2. **Segment** tumor regions for detailed analysis

**Instructions:**
1. Upload an MRI brain scan image
2. Click either 'Brain Detection' or 'Brain Segmentation'
3. View the results and analysis

**Supported formats:** JPG, JPEG, PNG, TIF
""")


