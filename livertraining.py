import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
df = pd.read_csv('indian_liver_patient.csv')

# Display the first few rows of the dataset
print(df.head())

# Encode categorical 'Gender' column
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Separate features and target label
X = df.drop(columns=['Dataset'])
y = df['Dataset']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert target labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=3)  # Assuming there are 3 classes, modify if needed
y_test_cat = to_categorical(y_test, num_classes=3)

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Assuming 3 output classes, change if necessary

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_cat, epochs=100, batch_size=32, validation_data=(X_test, y_test_cat))

# Save the model to an H5 file
model.save('liver_disease_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f'Loss: {loss}, Accuracy: {accuracy}')