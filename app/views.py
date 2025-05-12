from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names (example)
class_names = ['Parrot', 'Peacock', 'Sparrow', 'Crow', 'Eagle']

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Assuming input size for model
    input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(output_data)
    confidence = float(output_data[predicted_index]) * 100
    predicted_class = class_names[predicted_index]
    return predicted_class, round(confidence, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_class = None
    confidence = None
    full_filename = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            pred_class, confidence = predict_image(filepath)
            full_filename = f"uploads/{filename}"

    return render_template("index.html", pred_class=pred_class, confidence=confidence, full_filename=full_filename)

if __name__ == '__main__':
    app.run(debug=True)
