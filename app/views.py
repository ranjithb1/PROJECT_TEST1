# Important imports
from app import app
from flask import request, render_template
from keras import models
import numpy as np
from PIL import Image
import string
import random
import os

# Config path for image uploads
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Load the trained model
model = models.load_model('app/static/model/bird_species.h5')

# Class labels (make sure they are in the correct order of your model's output)
classes = [
    'AMERICAN GOLDFINCH',
    'BARN OWL',
    'CARMINE BEE-EATER',
    'DOWNY WOODPECKER',
    'EMPEROR PENGUIN',
    'FLAMINGO'
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Load default background image on GET
        full_filename = 'images/white_bg.jpg'
        return render_template("index.html", full_filename=full_filename)

    if request.method == "POST":
        # Generate unique file name
        name = ''.join(random.choices(string.ascii_lowercase, k=10)) + '.png'
        full_filename = 'uploads/' + name

        # Read uploaded image
        image_upload = request.files['image']
        image = Image.open(image_upload).convert('RGB')
        image = image.resize((224, 224))

        # Save the image
        image_path = os.path.join(app.config['INITIAL_FILE_UPLOADS'], name)
        image.save(image_path)

        # Preprocess image
        image_arr = np.array(image).reshape(1, 224, 224, 3) / 255.0

        # Make prediction
        result = model.predict(image_arr)
        ind = np.argmax(result)
        confidence = round(result[0][ind] * 100, 2)

        prediction = classes[ind]

        return render_template(
            'index.html',
            full_filename=full_filename,
            pred_class=prediction,
            confidence=confidence
        )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)