from app import app
from flask import request, render_template
import numpy as np
from PIL import Image
import string
import random
import os
import tensorflow as tf

app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Load model and class labels (keep as-is)
interpreter = tf.lite.Interpreter(model_path='app/static/model/bird_species_quant.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
classes = ['AMERICAN GOLDFINCH', 'BARN OWL', 'CARMINE BEE-EATER', 'DOWNY WOODPECKER', 'EMPEROR PENGUIN', 'FLAMINGO']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", full_filename='images/white_bg.jpg')
    
    if request.method == "POST":
        # (Keep existing image upload and prediction logic)
        name = ''.join(random.choices(string.ascii_lowercase, k=10)) + '.png'
        full_filename = 'uploads/' + name
        image_upload = request.files['image']
        image = Image.open(image_upload).convert('RGB').resize((224, 224))
        image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))
        
        # Run model inference (unchanged)
        image_arr = np.array(image).reshape(1, 224, 224, 3).astype(np.float32) / 255.0
        interpreter.set_tensor(input_details[0]['index'], image_arr)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])
        
        ind = np.argmax(result)
        return render_template('index.html', 
            full_filename=full_filename,
            pred_class=classes[ind],
            confidence=round(result[0][ind] * 100, 2)
        )

if __name__ == '__main__':
    app.run(debug=True)



