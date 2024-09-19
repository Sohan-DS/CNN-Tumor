import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load the trained CNN model
model = tf.keras.models.load_model('CNN/tumor_detection/results/model/cnn_tumor.h5')

# Function to make predictions
def make_prediction(image, model):
    img = Image.fromarray(image)
    img = img.resize((128, 128))  # Resizing the image to match model input size
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(input_img)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"

# Streamlit app title
st.title("Brain Tumor Detection App")

# Image uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the uploaded image to OpenCV format (numpy array)
    image = np.array(image)

    # Make a prediction
    result = make_prediction(image, model)

    # Display the result
    st.write(f"Prediction: {result}")