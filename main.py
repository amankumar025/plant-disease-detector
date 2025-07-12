# Load and preprocess images from a dataset of plant species
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Parameters
DATASET_DIR = "PlantVillage"
IMG_SIZE = 128

# Load images and labels
images = []
labels = []

for folder_name in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    for image_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(folder_name)
        except:
            continue

print("Total images:", len(images))

# Normalize images and convert labels to numpy arrays
# Convert to numpy array and normalize
X = np.array(images) / 255.0
y = np.array(labels)

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("Shape of X_train:", X_train.shape)
print("Number of classes:", len(np.unique(y_encoded)))
print("Class labels:", le.classes_)

# Build a simple CNN model using TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                validation_split=0.2)

# Save model and label encoder
model.save("model/plant_disease_model.h5")

import pickle
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Evaluate the model
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()


# predicting a new image
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Make it batch-like

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_name = le.inverse_transform([class_index])[0]

    print(f"Predicted Class: {class_name}")

# Test it on a sample image
predict_image("test_images/e1c5cd9c-d181-42b8-95cc-e45538f21d12___NREC_B.Spot 1864.JPG")
