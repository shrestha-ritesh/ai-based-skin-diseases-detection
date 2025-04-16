#!/usr/bin/env python
# coding: utf-8

# Main class for running the system.

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Class labels
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Sidebar selection for model type
model_type = st.sidebar.selectbox("Choose a model:", ["EfficientNetB0", "Xception"])

# Load the model
if model_type == "EfficientNetB0":
    model = load_model("models/efn_model.h5")
    print("Loded XCEPTION 500")
    input_size = (128, 128)
    preprocess_input = efficientnet_preprocess
else:
    model = load_model("models/xception_model_updated.h5")
    input_size = (128, 128)
    preprocess_input = xception_preprocess

# Title and description
st.title("Skin Disease Classification App")
st.write("Upload a dermatoscopic image to classify the lesion type using a deep learning model.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize(input_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]

    # Show result
    st.markdown(f"### Predicted Class: `{predicted_label}`")
    st.markdown("### Confidence Scores:")

    # Show a bar chart of the prediction
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction, color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Show raw scores
    for cls, prob in zip(class_names, prediction):
        st.write(f"{cls}: {prob:.4f}")



