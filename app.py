import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
from skimage import exposure
from skimage.restoration import denoise_nl_means
from skimage.transform import resize

# Load the pre-trained Keras model
model = tf.keras.models.load_model(r'D:\Semester3\Ishant\Final\Notebooks\best_cnn_model.h5')

# Preprocessing functions
def resize_and_normalize(image, desired_width, desired_height):
    resized_image = resize(image, (desired_height, desired_width))
    normalized_image = exposure.rescale_intensity(resized_image, out_range=(0, 255))
    return normalized_image

def reduce_noise(image):
    denoised_image = denoise_nl_means(image, patch_size=5, patch_distance=3, h=0.8)
    return denoised_image

def enhance_contrast(image, alpha=1.5, beta=0):
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image

def extract_roi(image):
    height, width = image.shape[:2]
    roi_width = int(width * 0.8)
    roi_height = int(height * 0.9)
    x = int((width - roi_width) / 2)
    y = int((height - roi_height) / 2)
    roi = image[y:y+roi_height, x:x+roi_width]
    return roi

# Function to preprocess the uploaded image
def preprocess_image(image: Image.Image):
    # Convert the image to grayscale
    image = image.convert("L")
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Apply the preprocessing steps
    resized_normalized_image = resize_and_normalize(image_array, 224, 224)
    denoised_image = reduce_noise(resized_normalized_image)
    contrast_enhanced_image = enhance_contrast(denoised_image)
    roi_image = extract_roi(contrast_enhanced_image)
    
    # Normalize to [0, 1] range
    roi_image = roi_image / 255.0
    roi_image = np.expand_dims(roi_image, axis=-1)  # Add channel dimension
    roi_image = np.expand_dims(roi_image, axis=0)  # Add batch dimension
    
    return roi_image

# Set up the Streamlit app
st.title('Bone Age Prediction App')

st.write("""
This app predicts the bone age from an X-ray image. Please upload an image file to get started.
""")

# Upload an image file
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Predict bone age using the model
    prediction = model.predict(preprocessed_image)
    # # Convert months to years
    # predicted_age_in_years = prediction[0][0] / 12
    # Convert the predicted age from months to years and months
    total_months = int(prediction[0][0])
    years = total_months // 12
    
    # Display the prediction result
    st.subheader('Predicted Bone Age:')
    # # Display the result in years
    # st.write(f'{predicted_age_in_years:.2f} years')
    # Display the result in "X years and Y months" format
    st.write(f'{years} years')

# Add a disclaimer in the sidebar
st.sidebar.header('Disclaimer')
st.sidebar.write("""
This app is for educational purposes only and should not be used for actual medical diagnosis. 
Always consult with a qualified healthcare professional for medical advice and diagnoses.
""")
