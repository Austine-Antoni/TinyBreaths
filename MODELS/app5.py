import os
import datetime
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from supabase import create_client
import serial

# Supabase connection
API_URL = 'https://ocrlmdadtekazfnhmquj.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jcmxtZGFkdGVrYXpmbmhtcXVqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1MTA2MzksImV4cCI6MjA1NzA4NjYzOX0.25bkWBV3v4cyjcA_-dUL8-IK3fSywARfVQ82UsZPelc'
supabase = create_client(API_URL, API_KEY)

# Load model and scaler
model_dir = "C:/Users/Austine/Desktop/MODELS/LSTM-v4"
model_path = os.path.join(model_dir, "LSTM_model.h5")
scaler_path = os.path.join(model_dir, "scaler.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

st.title("Real-Time Respiratory Monitoring")

# Create placeholders for live updates
live_count_placeholder = st.empty()
total_count_placeholder = st.empty()
chart_placeholder = st.empty()
data_table_placeholder = st.empty()

# Function to fetch the latest data from Supabase
def fetch_latest_data():
    response = supabase.table('maintable').select('*').order('timestamp', desc=True).limit(20).execute()
    if response.data:
        return response.data
    return []

# Function to make predictions
def predict_category(stored_count_60s):
    if stored_count_60s == 0:
        return None  # Do not predict if stored_count_60s is 0
    
    new_input = np.array([[60, stored_count_60s]])
    new_input_scaled = scaler.transform(new_input.reshape(-1, 2))
    new_input_reshaped = new_input_scaled.reshape(1, 1, 2)
    
    new_prediction = model.predict(new_input_reshaped)
    predicted_category = np.argmax(new_prediction, axis=1)
    
    category_map = ['Bradypnea', 'Normal', 'Tachypnea']
    return category_map[int(predicted_category[0])]

# Function to update Supabase with prediction
def update_supabase_prediction(record_id, prediction):
    if prediction is not None:
        response = supabase.table("maintable").update({"prediction": prediction}).eq("id", record_id).execute()
        if response.data:
            print(f"Updated prediction for ID {record_id}: {prediction}")

# Main loop to process data in real-time
while True:
    try:
        latest_data_list = fetch_latest_data()
        
        if latest_data_list:
            df = pd.DataFrame(latest_data_list)
            df["count"] = df["count"].astype(int)
            df["count_60s"] = df["count_60s"].astype(int)
            df["stored_count_60s"] = df["stored_count_60s"].astype(int)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            latest_data = df.iloc[0]
            
            if "prediction" not in latest_data or pd.isna(latest_data["prediction"]):
                predicted_value = predict_category(latest_data["stored_count_60s"])
                update_supabase_prediction(latest_data["id"], predicted_value)
                latest_data["prediction"] = predicted_value
            
            # Update UI with new data
            live_count_placeholder.metric("Live RR per minute", latest_data["count_60s"])
            total_count_placeholder.metric("Total RR", latest_data["count"])
            
            # Plot updated chart with unique key
            fig = px.line(df, x="timestamp", y=["count_60s", "count"], 
                          title=f"Respiratory Rate Over Time (Latest: {latest_data['timestamp']})",
                          labels={"timestamp": "Time", "count_60s": "RR per min", "count": "Total RR"})
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=str(datetime.datetime.now().timestamp()))
            data_table_placeholder.dataframe(df)
    except Exception as e:
        st.error(f"Error in main loop: {e}")
