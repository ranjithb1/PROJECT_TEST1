import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model("app/static/model/bird_species.h5")

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the quantized model
with open("app/static/model/bird_species_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Quantized model saved as bird_species_quant.tflite")
