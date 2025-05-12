from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max 5MB for images

# Load your pre-trained model
MODEL_PATH = 'path_to_your_trained_model.tflite'
model = tf.lite.Interpreter(model_path=MODEL_PATH)

# Utility function to check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict bird species
def predict_bird_species(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])  # Resize to model input size
    img = tf.cast(img, tf.float32) / 255.0  # Normalize image
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction using the model
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], img.numpy())
    model.invoke()

    # Get prediction result
    prediction = model.get_tensor(output_details[0]['index'])[0]
    predicted_class_idx = prediction.argmax()  # Get class with highest probability
    confidence = prediction[predicted_class_idx] * 100

    # Mapping the prediction index to class labels
    bird_classes = ["Class1", "Class2", "Class3", "Class4"]  # Example, replace with your actual classes
    predicted_class = bird_classes[predicted_class_idx]

    return predicted_class, confidence

# Home route to render the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get prediction result
        predicted_class, confidence = predict_bird_species(filepath)

        # Render the result in the webpage
        return render_template('index.html', 
                               full_filename=filename, 
                               pred_class=predicted_class, 
                               confidence=confidence)
    
    return "File type not allowed", 400

if __name__ == '__main__':
    app.run(debug=True)

