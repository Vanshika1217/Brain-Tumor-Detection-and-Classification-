from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)  # Fixed __name__
CORS(app)

# Load the model
model = load_model('model.h5')

# Set up the upload folder
basepath = os.path.dirname(__file__)  # Fixed __file__
upload_folder = os.path.join(basepath, 'uploads')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route('/')
def index():
    return render_template('index.html')

def process_image(img_path):
    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_class_name(class_no):
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return class_names[class_no]

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Process the image and make a prediction
        processed_image = process_image(file_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = get_class_name(predicted_class)

        return jsonify({'prediction': result})

if __name__ == '__main__':  # Fixed __name__
    app.run(debug=True, port=5500)
