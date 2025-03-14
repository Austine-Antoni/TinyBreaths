import os
import datetime
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from serial import Serial, SerialException

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

# Streamlit UI setup
st.title("Real-Time Respiratory Monitoring")

# Define counters
count = 0  # Total count
count_60s = 0  # Live count resets every 60 seconds
start_time = datetime.datetime.now()

# Create placeholders for live updates
live_count_placeholder = st.empty()
total_count_placeholder = st.empty()
stored_count_placeholder = st.empty()
chart_placeholder = st.empty()
data_table_placeholder = st.empty()

# List to store predictions and timestamps
data = []

# Function to establish serial connection
def connect_arduino():
    while True:
        try:
            arduino = Serial("COM11", 115200)
            print("Arduino connected successfully!")
            return arduino
        except SerialException:
            print("Arduino not detected. Retrying...")
            global count, count_60s, stored_count_60s, data, start_time
            count = 0
            count_60s = 0
            stored_count_60s = 0
            data.clear()
            start_time = datetime.datetime.now()

# Initialize Serial Communication
arduino = connect_arduino()

# Function to read data from Arduino
def grab_set():
    global count, count_60s, stored_count_60s, arduino
    try:
        arduino_data = arduino.readline().decode("utf-8").strip()
        
        if "Stored Count_60s:" in arduino_data:
            stored_count_60s = int(arduino_data.split(":")[1].strip())  # Fetch stored_count_60s directly
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predicted_category = predict_category(stored_count_60s)
            
            # Store data for visualization
            data.append({"Timestamp": timestamp, "Count_60s": stored_count_60s, "Prediction": predicted_category, "Count": count})
            
            # Convert to DataFrame and plot
            df = pd.DataFrame(data)
            fig = px.line(df, x="Timestamp", y=["Count_60s", "Count"], title="Respiratory Rate Over Time")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            data_table_placeholder.dataframe(df)
        elif "Count_60s:" in arduino_data:
            count_60s = int(arduino_data.split(":")[1].strip())
        elif "Count:" in arduino_data:
            count = int(arduino_data.split(":")[1].strip())
    except SerialException:
        print("Arduino disconnected. Reconnecting...")
        arduino = connect_arduino()
    except Exception as e:
        print(f"Error reading serial data: {e}")

# Function to make predictions
def predict_category(count_60s):
    new_input = np.array([[60, count_60s]])
    new_input_scaled = scaler.transform(new_input.reshape(-1, 2))
    new_input_reshaped = new_input_scaled.reshape(1, 1, 2)
    
    new_prediction = model.predict(new_input_reshaped)
    predicted_category = np.argmax(new_prediction, axis=1)
    
    category_map = ['Bradypnea', 'Normal', 'Tachypnea']
    predicted_category = category_map[int(predicted_category[0])]
    
    return predicted_category

# Main loop to process data
while True:
    try:
        grab_set()  # Read from Arduino
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Display live values
        live_count_placeholder.metric("Live RR per minute", count_60s)
        total_count_placeholder.metric("Total RR", count)
    except Exception as e:
        print(f"Error in main loop: {e}")
