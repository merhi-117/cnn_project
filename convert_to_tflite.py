import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("model.keras")

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations for smaller, faster models
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Converted model saved as model.tflite")
