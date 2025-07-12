import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# Constants
IMG_SIZE = 128

# Load model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Load label encoder
with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Title and Upload
st.title(" Plant Disease Detector")
st.write("Upload a leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_class = le.inverse_transform([predicted_index])[0]

    st.success(f" Predicted Disease: **{predicted_class}**")

# save the model and label encoder
# Save model
model.save("model/plant_disease_model.h5")

# Save label encoder
import pickle
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
