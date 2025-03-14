import os
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load model and scaler
model_dir = "C:/Users/Austine/Desktop/MODELS/LSTM-v4"
model_path = os.path.join(model_dir, "LSTM_model.h5")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# Load LSTM model
model = load_model(model_path)
print("Model loaded successfully!")

# Load Scaler
scaler = joblib.load(scaler_path)
print("Scaler loaded successfully!")

# New input (60 seconds, respiratory rate of 60)
new_input = np.array([[60, 30]])  # First feature = timestamp (60), second feature = count_60s

# Normalize using the same scaler used during training
new_input_scaled = scaler.transform(new_input.reshape(-1, 2))

# Reshape for LSTM input
new_input_reshaped = new_input_scaled.reshape(1, 1, 2)  # Shape: (1, time_steps, 2)

# Make the prediction
new_prediction = model.predict(new_input_reshaped)
predicted_category = np.argmax(new_prediction, axis=1)

# Map to category labels
category_map = ['Bradypnea', 'Normal', 'Tachypnea']
predicted_category = category_map[int(predicted_category[0])]

print(f'Predicted Category: {predicted_category}')
