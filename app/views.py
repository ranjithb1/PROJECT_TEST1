import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Path to the .tflite model
MODEL_PATH = 'app/static/model/bird_species_quant.tflite'

# Load the model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust this size based on your model's input size
    img = np.array(img) / 255.0  # Normalize if required by your model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make prediction using the model
def predict(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    
    return predicted_class, output_data[0][predicted_class]

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict_bird_species():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Save the image file temporarily
            image_path = os.path.join('app/static/uploads', file.filename)
            file.save(image_path)

            # Preprocess the image and make prediction
            processed_image = preprocess_image(image_path)
            predicted_class, confidence = predict(processed_image)

            # Send response
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': str(confidence),
                'message': 'Prediction successful!'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)

