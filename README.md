# Plant Disease Detection AI App

This is a deep learning-powered web app that detects plant leaf diseases from images using a Convolutional Neural Network (CNN). The app is built with **TensorFlow**, **OpenCV**, and deployed using **Streamlit**.

![App Screenshot](https://via.placeholder.com/800x400.png?text=Add+your+Streamlit+app+screenshot+here)

---

## Features

- Upload leaf images directly through the web interface
- Predicts plant disease with high accuracy
- Uses real-world PlantVillage dataset
- Built using:
  - TensorFlow & Keras
  - OpenCV
  - Streamlit
  - Scikit-learn

---

## Model Summary

- Trained on 10+ plant disease classes
- Architecture: 3-layer CNN
- Image input size: 128x128
- Achieved ~90% validation accuracy

---

## Project Structure
plant_disease_detector/
├── model/
│ ├── plant_disease_model.h5 # Trained model file
│ ├── label_encoder.pkl # Encoded labels
├── streamlit_app.py # Web app
├── main.py # Training code
├── requirements.txt # Dependencies
├── test_images/ # Example test images
├── .gitignore
└── README.md


---

##  Installation

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/plant-disease-detector.git
cd plant-disease-detector

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate    # On Windows
# or
source venv/bin/activate  # On Mac/Linux

# Install required packages
pip install -r requirements.txt

streamlit run streamlit_app.py
