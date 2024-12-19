from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_image(model, filepath):
    # Load and preprocess the image
    image = load_img(filepath, target_size=(128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Make prediction
    prediction = model.predict(image)
    print("Prediction Raw Output:", prediction)

    # Assuming the model outputs probabilities for two classes: [prob_uninfected, prob_parasitized]
    if prediction.shape[1] == 2:
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        result = 'Parasitized' if predicted_class == 1 else 'Uninfected'
    else:
        predicted_class = (prediction[0][0] > 0.5).astype("int32")
        confidence = prediction[0][0]
        result = 'Parasitized' if predicted_class == 1 else 'Uninfected'
    
    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)
    return result, confidence

# Load the model
model = load_model('malaria_detection_model_corrected.h5')

# Test with a known infected image
test_image_path = './cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_163.png'
result, confidence = predict_image(model, test_image_path)
print("Result:", result)
print("Confidence:", confidence)
