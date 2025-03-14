import os
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

# Streamlit UI setup
st.title("Real-Time Respiratory Monitoring")

# Define counters
count = 0  # Total count
count_60s = 0  # Live count resets every 60 seconds
start_time = datetime.datetime.now()

# Create placeholders for live updates
live_count_placeholder = st.empty()
total_count_placeholder = st.empty()
chart_placeholder = st.empty()
data_table_placeholder = st.empty()

# List to store predictions and timestamps
data = []

# Function to read data from Arduino
def grab_set():
    global count, count_60s
    try:
        arduino_data = arduino.readline().decode("utf-8").strip()
        if "Count:" in arduino_data:
            count = int(arduino_data.split(":")[1].strip())
            count_60s += 1  # Increment count_60s in real-time
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
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        grab_set()  # Read from Arduino
        
        # Display live values
        live_count_placeholder.metric("Live RR per minute", count_60s)
        total_count_placeholder.metric("Total RR", count)
        
        # Every 60 seconds, record the prediction
        if elapsed_time >= 60:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predicted_category = predict_category(count_60s)
            
            # Store data for visualization
            data.append({"Timestamp": timestamp, "Count_60s": count_60s, "Prediction": predicted_category})
            
            # Convert to DataFrame and plot
            df = pd.DataFrame(data)
            fig = px.line(df, x="Timestamp", y="Count_60s", title="Respiratory Rate Over Time")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            data_table_placeholder.dataframe(df)
            
            # Reset live count and restart timer
            count_60s = 0
            start_time = datetime.datetime.now()
    except Exception as e:
        print(f"Error in main loop: {e}")
