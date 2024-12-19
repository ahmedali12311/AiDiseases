import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# Paths to training directory
train_dir = './cell_images'

# Image dimensions
img_width, img_height = 128, 128

# Batch size
batch_size = 32

# Function to compute class weights to handle class imbalance
def compute_correct_class_weights(train_dir, class_indices):
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    
    total_samples = sum(class_counts.values())
    class_weights = {class_indices[class_name]: total_samples / (len(class_counts) * count) 
                     for class_name, count in class_counts.items()}
    
    return class_weights

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Check the class indices
class_indices = train_generator.class_indices
print("Class Indices: ", class_indices)

# Compute the correct class weights based on class indices
correct_class_weights = compute_correct_class_weights(train_dir, class_indices)
print("Correct Class Weights: ", correct_class_weights)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the correct class weights
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    class_weight=correct_class_weights
)

# Save the corrected model
model.save('malaria_detection_model_corrected.h5')

# Evaluate the corrected model on the validation set
validation_generator.reset()
eval_result = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Corrected Validation Loss: {eval_result[0]}, Corrected Validation Accuracy: {eval_result[1]}")
