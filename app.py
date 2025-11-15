import os
import numpy as np
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle
MODEL_PATH = 'best_malaria_model.h5'
model = load_model(MODEL_PATH)

IMG_SIZE = (64, 64)
class_labels = ['Parasitized', 'Uninfected']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = load_img(filepath, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_labels[int(np.round(prediction[0][0]))]

        return render_template('index.html',
                               prediction=predicted_class,
                               image_path=filepath)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render fournit PORT
    app.run(host="0.0.0.0", port=port, debug=True)
