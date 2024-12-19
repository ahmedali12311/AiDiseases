from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and set up the scaler
model = load_model('kidney_disease_model.h5')
scaler = StandardScaler()

# Sample data to fit the scaler, based on training data distribution
# Replace this with actual training data means and stds, if available
scaler.fit(np.random.rand(400, 24))  # Assuming 24 features after encoding

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data and convert it to a DataFrame
        form_data = request.form.to_dict()
        input_data = pd.DataFrame([form_data], columns=form_data.keys())

        # Convert numerical values and encode categorical values as required
        # Here we assume preprocessing similar to training; modify accordingly
        input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Scale the data
        input_data = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data)
        prediction_class = "Sick" if prediction[0] < 0.5 else "Not Sick"

        # Return the prediction
        return jsonify({"prediction": prediction_class})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
