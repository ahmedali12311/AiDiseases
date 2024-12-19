import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('kidney_disease_model.h5')

# Define the test data
# Sick Patient (CKD)
sick_patient = np.array([[240, 65.0, 70.0, 1.015, 1.0, 0.0, 0,  # Packed Cell Volume (missing replaced with 0)
                          0, 0, 203.0, 46.0, 1.4, 0, 0, 11.4, 36, 5000, 4.1, 1, 1, 0,  # Hypertension, Diabetes Mellitus, etc.
                          1, 1, 1]])  # Poor appetite, pedal edema, anaemia

# Not Sick Patient (Not CKD)
not_sick_patient = np.array([[250, 40.0, 80.0, 1.025, 0.0, 0.0, 0,  # Packed Cell Volume (missing replaced with 0)
                              0, 0, 140.0, 10.0, 1.2, 135.0, 5.0, 15.0, 48, 10400, 4.5, 0, 0, 0,  # Hypertension, Diabetes Mellitus, etc.
                              1, 0, 0]])  # Good appetite, no pedal edema, no anaemia

# Combine the test data
test_data = np.vstack((sick_patient, not_sick_patient))

# Standardize the test data
scaler = StandardScaler()
# Note: Make sure to fit the scaler on the training data and save it for consistent transformation
# Here we assume the scaler is already fitted on the training data
# If you have the scaler saved, load it instead of fitting again
# scaler = joblib.load('scaler.pkl')  # Example of loading a saved scaler
test_data_scaled = scaler.fit_transform(test_data)

# Make predictions
predictions = model.predict(test_data_scaled)

# Output the predictions
for i, prediction in enumerate(predictions):
    print(f"Patient {i + 1} prediction: {'CKD' if prediction[0] >= 0.5 else 'Not CKD'}, Probability: {prediction[0]}")