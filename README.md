# image-classification
image classification ml model
Image Classification using Convolutional Neural Networks (CNN)

A multi-class image classification system built using Convolutional Neural Networks (CNN) in TensorFlow/Keras.
The model classifies images into four different categories and is designed as an academic machine learning mini-project.

📌 Project Overview

This project demonstrates how deep learning can be used to automatically classify images into predefined categories.
A CNN model is trained on image datasets and later used to predict the class of new images.

🧠 Classes Supported

The model classifies images into the following four categories:

Cat vs Dog

Cat

Multi-object

Traffic

⚙️ Technologies Used

Python 🐍

TensorFlow & Keras

Convolutional Neural Networks (CNN)

NumPy

Matplotlib

Pillow (PIL)

Jupyter Notebook

📂 Project Structure
Image_Classification_Project/
│
├── IMAGE_CLASSIFICATION.ipynb    # Model training & testing
├── 4_class_image_classifier.h5  # Trained CNN model
├── dataset/                      # Training & testing images
│   ├── train/
│   └── test/
├── sample.jpg                    # Test image for prediction
└── README.md                     # Project documentation

🧠 How the Model Works

Images are resized and normalized

CNN extracts spatial features using convolution layers

Max-Pooling reduces spatial dimensions

Fully connected layers perform classification

Softmax activation outputs class probabilities

▶️ How to Run the Project
Step 1: Install Required Libraries
pip install tensorflow numpy matplotlib pillow

Step 2: Open Jupyter Notebook
jupyter notebook

Step 3: Run the Notebook

Open IMAGE_CLASSIFICATION.ipynb and run all cells:

Dataset loading

Model training

Model saving

Image prediction

🧪 Sample Prediction Code
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("4_class_image_classifier.h5")

img = image.load_img("sample.jpg", target_size=(96,96))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_names = ["catvsdog", "cat", "multi", "trafic"]

print("Prediction:", class_names[np.argmax(prediction)])

📊 Results

Successfully classifies images into 4 categories

Achieves good accuracy on test images

Robust to noisy or unseen images

🎓 Learning Outcomes

Understanding CNN architecture

Image preprocessing techniques

Model training & evaluation

Saving and loading trained models

Real-world image classification workflow

🚀 Future Enhancements

Improve accuracy using transfer learning (MobileNet, VGG)

Build a web application using Streamlit

Add real-time camera image classification

Increase dataset size
