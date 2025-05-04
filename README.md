# MNIST Handwritten Digit Classifier (TensorFlow + TFLite)

This project showcases a full machine learning pipeline from training a Convolutional Neural Network (CNN) model using TensorFlow on the MNIST dataset, to converting it into a TensorFlow Lite model ready for deployment on edge devices such as Android, iOS, or Raspberry Pi.

---

## 📌 Project Goals

- Build a CNN that can accurately classify handwritten digits (0–9) from grayscale images.
- Train and evaluate the model using the MNIST dataset.
- Convert the trained model to `.tflite` format for edge deployment.
- Demonstrate predictions using both the full `.keras` model and the `.tflite` model.

---

## 🧠 Model Overview

- **Architecture**: Conv2D layers with ReLU activations, MaxPooling, Dropout, and Dense layers.
- **Dataset**: [MNIST](https://en.wikipedia.org/wiki/MNIST_database) – 60,000 training and 10,000 test images.
- **Accuracy Achieved**: ~98% on test set.

---

## 📂 Repository Structure

cnn_projct/
│
├── mnist_cnn.py # CNN training script using TensorFlow/Keras
├── convert_to_tflite.py # Converts trained .keras model to TensorFlow Lite
├── predict_mnist.py # Uses the .keras model for prediction
├── predict_tflite.py # Uses the .tflite model for prediction (edge focus)
│
├── model.keras # Saved model from training
├── model.tflite # Optimized model for edge deployment
├── model.png # (Optional) CNN architecture visualization
│
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions

---

## 🚀 How to Run

### 1. Setup Environment

pip install -r requirements.txt

### 2. Train the CNN

python mnist_cnn.py

### 3. Convert to TFLite

python convert_to_tflite.py

### 4. Make Predictions

python predict_mnist.py
python predict_tflite.py

📱 Edge Deployment Ready
The .tflite model is compatible with:

- Android and iOS apps using TensorFlow Lite
- Edge devices like Raspberry Pi

## 🧑‍💻 Author  
**Alaa Merhi**  
BSc in Computer Science, York University  
📫 [Alaamerhi117@gmail.com](mailto:Alaamerhi117@gmail.com)  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/alaa-merhi)

