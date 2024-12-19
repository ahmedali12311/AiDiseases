import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
data = pd.read_csv('kidney_disease.csv')

# Print unique values in the 'classification' column before mapping
print("Unique values in 'classification' before mapping:")
print(data['classification'].unique())

# Clean the data: Replace non-numeric entries with NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Fill NaN values (you can choose a different method if needed)
data.fillna(method='ffill', inplace=True)

# Convert categorical variables to numerical
data['rbc'] = data['rbc'].map({'normal': 0, 'abnormal': 1})
data['pc'] = data['pc'].map({'normal': 0, 'abnormal': 1})
data['pcc'] = data['pcc'].map({'notpresent': 0, 'present': 1})
data['ba'] = data['ba'].map({'notpresent': 0, 'present': 1})
data['htn'] = data['htn'].map({'no': 0, 'yes': 1})
data['dm'] = data['dm'].map({'no': 0, 'yes': 1})
data['cad'] = data['cad'].map({'no': 0, 'yes': 1})
data['appet'] = data['appet'].map({'good': 0, 'poor': 1})
data['pe'] = data['pe'].map({'no': 0, 'yes': 1})
data['ane'] = data['ane'].map({'no': 0, 'yes': 1})

# Map the target variable
data['classification'] = data['classification'].map({'ckd': 0, 'notckd': 1})

# Print unique values in the 'classification' column after mapping
print("Unique values in 'classification' after mapping:")
print(data['classification'].unique())

# Check for NaN values in the target variable
print("NaN values in target variable before dropping:")
print(data['classification'].isnull().sum())

# Drop rows with NaN values in the target variable
data.dropna(subset=['classification'], inplace=True)

# Features and target variable
X = data.drop(columns=['id', 'classification'])
y = data['classification']

# Check if y is empty after dropping NaNs
if y.empty:
    print("Target variable 'y' is empty after dropping NaNs!")
else:
    print("Target variable 'y' has valid entries.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# Save the model
model.save('kidney_disease.h5')

print("Model saved as kidney_disease.h5")