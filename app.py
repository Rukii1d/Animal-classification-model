import os
import numpy as np
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf

# Streamlit app header
st.header('Animal Classification CNN Model')

# Animal names for classification
animal_names = ['Bird', 'Cat', 'Dog', 'Elephant', 'Snake']

# Update the path to your model file here
model_path = 'Animal_Rec_Model.h5'  # Replace with the actual path to your model

# Check if the model file exists
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"Model file not found at {model_path}. Please check the path.")
    st.stop()

# Function to classify uploaded image
def classify_image(image):
    input_image = tf.keras.utils.load_img(image, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    
    # Prepare the result
    outcome = 'Our Model Says It Is: ' + animal_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100) + '%'
    return outcome

# Streamlit file uploader to upload an image
uploaded_image = st.file_uploader("Upload an Image of an Animal", type=["jpg", "png", "jpeg"])

# If an image is uploaded, classify it
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)  # Corrected here
    st.write("")  # Adding a little space
    st.write("Classifying...")

    # Call the classification function
    result = classify_image(uploaded_image)
    st.write(result)
