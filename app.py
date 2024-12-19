import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import locale
from io import BytesIO
from PIL import Image
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set the locale to ensure UTF-8 encoding
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_AS_ASCII'] = False  # Ensures JSON responses use UTF-8 encoding

# Load models
malaria_model = load_model('malaria_detection_model_corrected.h5')
diabetes_model = load_model('project_model.h5')
liver_disease_model = load_model('liver_disease_model.h5')  # Load liver disease model
anemia_model = load_model('cbc_model.h5')

# Load the label encoder and scaler for anemia prediction
with open('label_encoder.pkl', 'rb') as f:
    anemia_label_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image):
    image = image.resize((128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    prediction = malaria_model.predict(image)
    return 'غير مصاب' if prediction[0][0] > 0.5 else 'مصاب'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/malaria', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image = Image.open(file)
            prediction = predict_image(image)
            return render_template('malariaresult.html', prediction=prediction)
    return render_template('malaria.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        polyuria = int(request.form['polyuria'])
        polydipsia = int(request.form['polydipsia'])
        sudden_weight_loss = int(request.form['sudden_weight_loss'])
        weakness = int(request.form['weakness'])
        polyphagia = int(request.form['polyphagia'])
        genital_thrush = int(request.form['genital_thrush'])
        visual_blurring = int(request.form['visual_blurring'])
        itching = int(request.form['itching'])
        irritability = int(request.form['irritability'])
        delayed_healing = int(request.form['delayed_healing'])
        partial_paresis = int(request.form['partial_paresis'])
        muscle_stiffness = int(request.form['muscle_stiffness'])
        alopecia = int(request.form['alopecia'])
        obesity = int(request.form['obesity'])

        input_data = np.array([[age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, 
                                genital_thrush, visual_blurring, itching, irritability, delayed_healing, 
                                partial_paresis, muscle_stiffness, alopecia, obesity]])

        prediction = diabetes_model.predict(input_data)
        result = 'موجبة' if prediction[0] >= 0.5 else 'سالبة'

        return render_template('diabetes_result.html', prediction=result)
    return render_template('diabetes_form.html')

@app.route('/anemia', methods=['GET', 'POST'])
def predict_anemia():
    if request.method == 'POST':
        form_data = request.form
        
        try:
            # Collect features from the form
            features = [
                float(form_data.get('WBC', 0)),
                float(form_data.get('LYMp', 0)),
                float(form_data.get('NEUTp', 0)),
                float(form_data.get('LYMn', 0)),
                float(form_data.get('NEUTn', 0)),
                float(form_data.get('RBC', 0)),
                float(form_data.get('HGB', 0)),
                float(form_data.get('HCT', 0)),
                float(form_data.get('MCV', 0)),
                float(form_data.get('MCH', 0)),
                float(form_data.get('MCHC', 0)),
                float(form_data.get('PLT', 0)),
                float(form_data.get('PDW', 0)),
                float(form_data.get('PCT', 0)),
            ]
        except ValueError as ve:
            return f"Error converting form data to numbers: {ve}"

        # Convert features to a DataFrame
        df = pd.DataFrame([features], columns=[
            'WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 
            'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT'
        ])

        # Scale the features using the loaded scaler
        X_scaled = scaler.transform(df)

        # Make predictions
        prediction = anemia_model.predict(X_scaled)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = anemia_label_encoder.inverse_transform(predicted_class)

        return render_template('anemia_result.html', prediction=predicted_label[0])

    return render_template('anemia_form.html')
@app.route('/liver', methods=['GET', 'POST'])
def predict_liver_disease():
    if request.method == 'POST':
        # Retrieve form data
        try:
            age = float(request.form['age'])
            gender = 1 if request.form['gender'] == 'Male' else 0  # Assuming Male=1, Female=0
            total_bilirubin = float(request.form['total_bilirubin'])
            direct_bilirubin = float(request.form['direct_bilirubin'])
            alkaline_phosphotase = float(request.form['alkaline_phosphotase'])
            alamine_aminotransferase = float(request.form['alamine_aminotransferase'])
            aspartate_aminotransferase = float(request.form['aspartate_aminotransferase'])
            total_proteins = float(request.form['total_proteins'])
            albumin = float(request.form['albumin'])
            albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio'])
        except ValueError as ve:
            return f"Error converting form data to numbers: {ve}"

        # Prepare the input data as a NumPy array
        input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                                albumin, albumin_and_globulin_ratio]])

        # Scale the input data using the same scaler used during training
        input_data_scaled = scaler.transform(input_data)

        # Predict using the loaded model
        prediction = liver_disease_model.predict(input_data_scaled)

        # Since the prediction is a 2D array, we access the first element to compare it with 0.5
        result = 'Liver Disease Present' if prediction[0][0] >= 0.5 else 'No Liver Disease'

        return render_template('liver_result.html', prediction=result)

    return render_template('liver_form.html')

if __name__ == '__main__':
    app.run(debug=True)