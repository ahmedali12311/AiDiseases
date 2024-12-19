from tensorflow import keras

# Load the .h5 model
model = keras.models.load_model('project_model.h5')

# Save the model in Keras format
model.save('exported_model.keras')  # Specify the filename with .keras extension