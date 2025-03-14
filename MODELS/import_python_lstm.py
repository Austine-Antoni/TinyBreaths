import os
import time
import datetime
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from serial import Serial

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

# Initialize Serial Communication
arduino = Serial("COM11", 9600)
time.sleep(2)  # Allow connection to establish

def grab_set():
    """Reads data from Arduino, extracts count_60s, and returns it."""
    try:
        arduino_data = arduino.readline().decode("utf-8").strip()
        if "Strains in last 60s:" in arduino_data:
            count_60s = arduino_data.split(":")[1].strip()  # Extract value
            if count_60s.isdigit():  # Ensure valid number
                count_60s = int(count_60s)
                print(f"count_60s: {count_60s}")
                return count_60s
    except Exception as e:
        print(f"Error reading serial data: {e}")
    return None  # Return None if no valid data

def predict_category(count_60s):
    """Predicts respiratory condition based on count_60s."""
    new_input = np.array([[60, count_60s]])  # First feature: 60s, Second: count_60s
    new_input_scaled = scaler.transform(new_input.reshape(-1, 2))
    new_input_reshaped = new_input_scaled.reshape(1, 1, 2)  # Reshape for LSTM input
    
    new_prediction = model.predict(new_input_reshaped)
    predicted_category = np.argmax(new_prediction, axis=1)
    
    # Map to category labels
    category_map = ['Bradypnea', 'Normal', 'Tachypnea']
    predicted_category = category_map[int(predicted_category[0])]
    
    print(f"count_60s: {count_60s} -> Predicted Category: {predicted_category}")
    print("Raw Model Output:", new_prediction)
    
    return predicted_category

# Real-time continuous prediction
print("Waiting for count_60s updates...")
while True:
    try:
        count_60s = grab_set()  # Read from Arduino
        if count_60s is not None:
            predict_category(count_60s)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        break
    except Exception as e:
        print(f"Error in main loop: {e}")
