import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret_key_123"   # Nécessaire pour la session (login)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------- Charger le modèle ----------
MODEL_PATH = 'best_malaria_model.h5'
model = load_model(MODEL_PATH)

IMG_SIZE = (64, 64)
class_labels = ['Parasitized', 'Uninfected']


# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')


# ----------- LOGIN PAGE -----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        pw = request.form['password']

        # LOGIN SIMPLE (tu pourras connecter une DB plus tard)
        if username == "admin" and pw == "123":
            session['user'] = username
            return redirect(url_for('predict_page'))
        else:
            return render_template('login.html', error="Identifiants incorrects")

    return render_template('login.html')


# ----------- LOGOUT -----------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ----------- PAGE PREDICTION (protégée) -----------
@app.route('/predict_page')
def predict_page():
    if "user" not in session:
        return redirect(url_for('login'))

    return render_template('predict.html')


# ----------- SUBMIT IMAGE + PREDICTION -----------
@app.route('/predict', methods=['POST'])
def predict():
    if "user" not in session:
        return redirect(url_for('login'))

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

        prediction = model.predict(img_array)[0][0]
        predicted_class = class_labels[int(np.round(prediction))]

        return render_template('predict.html',
                               prediction=predicted_class,
                               image_path=filepath)

# ----------- RUN SERVER -----------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
