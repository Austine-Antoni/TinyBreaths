import os
import time
import datetime
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
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

# Streamlit setup
st.title("Real-Time Respiratory Monitoring")
st.subheader("Live Count and Predictions")

# UI Elements
count_display = st.metric(label="RR Total", value=0)
count_60s_display = st.metric(label="RR per minute", value=0)
prediction_text = st.empty()

# Data storage
data_list = []  # Store real-time data for visualization
prev_minute = datetime.datetime.now().minute
count = 0
count_60s = 0

# Function to read data from Arduino
def grab_set():
    """Reads data from Arduino, extracts count and count_60s."""
    global count, count_60s
    try:
        arduino_data = arduino.readline().decode("utf-8").strip()
        if "Count:" in arduino_data:
            count = int(arduino_data.split(":")[1].strip())
        elif "Strains in last 60s:" in arduino_data:
            count_60s = int(arduino_data.split(":")[1].strip())
    except Exception as e:
        print(f"Error reading serial data: {e}")

# Function to make predictions
def predict_category(count_60s):
    """Predicts respiratory condition based on count_60s."""
    new_input = np.array([[60, count_60s]])
    new_input_scaled = scaler.transform(new_input.reshape(-1, 2))
    new_input_reshaped = new_input_scaled.reshape(1, 1, 2)
    
    new_prediction = model.predict(new_input_reshaped)
    predicted_category = np.argmax(new_prediction, axis=1)
    
    category_map = ['Bradypnea', 'Normal', 'Tachypnea']
    predicted_category = category_map[int(predicted_category[0])]
    
    return predicted_category

# Main loop
while True:
    try:
        current_minute = datetime.datetime.now().minute
        grab_set()  # Read from Arduino

        # Update live display
        count_display.metric(label="RR Total", value=count)
        count_60s_display.metric(label="RR per minute", value=count_60s)

        # Every minute, record prediction and reset count_60s
        if current_minute != prev_minute:
            prev_minute = current_minute
            predicted_category = predict_category(count_60s)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prediction_text.write(f"**Predicted Category:** {predicted_category} | RR per minute: {count_60s} | Timestamp: {timestamp}")
            data_list.append({"Timestamp": timestamp, "Count_60s": count_60s, "Prediction": predicted_category})
            count_60s = 0  # Reset count_60s every minute

        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        break
    except Exception as e:
        print(f"Error in main loop: {e}")
