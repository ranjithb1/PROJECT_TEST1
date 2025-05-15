  Bird Species Classifier using Transfer Learning (VGG16)

✅ Leverage VGG16 as a feature extractor by adding custom classification layers on top.
✅ Preprocess and augment image data using Keras' ImageDataGenerator to improve model robustness.
✅ Split the dataset into training, validation, and test sets for proper model evaluation.
✅ Train the model with callbacks such as early stopping and learning rate reduction to prevent overfitting.
✅ Evaluate model performance using metrics like accuracy, confusion matrix, and classification report.
✅ Visualize predictions by displaying actual vs predicted classes on random test images.
✅ Save the trained model in .h5 format for reuse or further training.
✅ Quantize and convert the model to .tflite with int16 optimization for faster inference and smaller size, ideal for Android/Web deployment.

🚀 Local Deployment using Flask
1. After training and quantizing the model, we implemented a lightweight web interface to run the bird species prediction locally using:
2. Flask – to serve the model and handle image uploads
3 .NumPy – for numerical operations and preprocessing
4. TensorFlow / Keras – to load the trained model
5. Pillow (PIL) – to handle image input and resizing
6. Gunicorn – for running the Flask server in production-ready environments

DEMO 
1. Model Information
2. Trained model: bird_species.h5
3. Input size: 224x224 RGB images 

Classes:
American Goldfinch
Barn Owl
Carmine Bee-Eater
Downy Woodpecker
Emperor Penguin
Flamingo

✨ Features
1. Upload any bird image and get instant prediction
2. Animated background using Vanta.js (Birds)
3. Elegant UI with glassmorphism and custom CSS
4. Displays confidence percentage for prediction

📸 How to Use
1. Clone the repository:
2. git clone https://github.com/ranjithbq/PROJECT_TEST1.git
3. python -m venv venv
4. source venv/bin/activate   # On Windows: venv\Scripts\activate

Install dependencies:
1. pip install -r requirements.txt
2. Run the Flask app:
3. python app/views.py
4. Open your browser and go to http://127.0.0.1:5000/.


🌐 Deployment
1. After developing the app locally, we deployed it to a live website using Render:
2. Pushed the project to a GitHub repository.
3. Created a new Web Service on Render.
4. Connected the GitHub repository to Render.
5. Specified the build command and start command:
6. Build Command: (leave blank or use pip install -r requirements.txt if needed)
7. Start Command: python app/views.py
8. Selected Python as the environment.
9. Deployed the app.

✅ After successful deployment, the web app is live and accessible on the internet.

✨ You can try it here: https://project-test1-xmti.onrender.com

📱 Android App Conversion
1. After deploying the website, we converted it into an Android app using Appilix:
2. Updated the HTML, CSS, and JavaScript for better Android WebView compatibility:
3. Ensured responsive design.
4. Enabled camera/image upload features to work in mobile browsers.
5. Used Appilix to convert the live website into an Android app:
6. Entered the deployed website URL.
7. Customized the app name, icon, and splash screen.
8. Generated and downloaded the .apk file.

✅ The Android app is now ready to install and use!

📦 Download the APK here: https://storage.appilix.com/uploads/app-apk-6824fae17f5fc-1747253985.apk

🚀 App Workflow and Features
1. Once the app is opened on an Android device:
2. The app connects to the Render-hosted backend and loads the web interface.
3. Users can upload an image in two ways:
4. By capturing a photo using the camera.
5. By selecting an existing file using the choose file option.
6. After uploading, the app:
7. Shows a preview of the selected image.
8. Offers a "Predict" button.
9. Upon clicking Predict: The image is processed by the deployed TensorFlow Lite model.
10. The app displays: The predicted bird species name.
11. The confidence score.
12. A link to more information about the bird.

✅ This makes the app fully functional for real-time bird species recognition!
