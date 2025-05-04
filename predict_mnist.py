import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("model.keras")

# Load MNIST data again (only test set is needed here)
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess test images
x_test = x_test / 255.0  # Normalize
x_test = x_test.reshape(-1, 28, 28, 1)  # Reshape for CNN input

# Predict the first 5 test images
predictions = model.predict(x_test[:5])

# Plot predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()
