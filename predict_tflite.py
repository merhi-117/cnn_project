import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Load and preprocess test data
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Predict and visualize first 5 test samples
    for i in range(5):
        img = np.expand_dims(x_test[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output)
        confidence = np.max(output)

        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {predicted} ({confidence:.2f})\nActual: {y_test[i]}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
