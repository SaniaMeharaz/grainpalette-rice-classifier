from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session

# Upload configuration
UPLOAD_FOLDER = os.path.join('static', 'uploaded')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("rice_model.h5")
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return "No file uploaded.", 400

        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ✅ Resize only for model — not for display
        img_model = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img_model)
        img_array = tf.expand_dims(img_array, 0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]) * 100)

        session['predicted_class'] = predicted_class
        session['confidence'] = round(confidence, 2)
        session['image_url'] = f'uploaded/{file.filename}'

        return redirect(url_for('result'))

    return render_template('predict.html')

@app.route('/result')
def result():
    if 'predicted_class' not in session:
        return redirect(url_for('predict'))

    return render_template('result.html',
                           predicted_class=session['predicted_class'],
                           confidence=session['confidence'],
                           image_url=session['image_url'])

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
