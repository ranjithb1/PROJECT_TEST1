from app import app
import numpy as np
from PIL import Image
import string
import random
import tensorflow as tf

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from predict import prediction

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limit upload to 5MB

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_class = None
    full_filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pred_class = prediction(filepath)
            full_filename = os.path.join('uploads', filename)  # Relative to /static/
    return render_template('index.html', pred_class=pred_class, full_filename=full_filename)

@app.after_request
def add_header(response):
    # Prevent caching and ensure compatibility with Android WebView
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True)
