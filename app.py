import os
import datetime
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from supabase import create_client

# Supabase connection
API_URL = 'https://ocrlmdadtekazfnhmquj.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jcmxtZGFkdGVrYXpmbmhtcXVqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1MTA2MzksImV4cCI6MjA1NzA4NjYzOX0.25bkWBV3v4cyjcA_-dUL8-IK3fSywARfVQ82UsZPelc'  
supabase = create_client(API_URL, API_KEY)

model = load_model("LSTM_model.h5")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Real-Time Respiratory Rate (RR) Monitoring")

# Placeholder for Date and Time
datetime_placeholder = st.subheader("📅 Loading date and time...")

# Layout: Table on the left, Readings & Alerts on the right
col1, col2 = st.columns([2, 1.5])

with col1:
    st.subheader("📋 Patient Chart")
    data_table_placeholder = st.empty()

with col2:
    st.subheader("📊 RR")
    live_count_placeholder = st.empty()
    total_count_placeholder = st.empty()
    status_placeholder = st.empty()  # Status for normal/warnings

st.subheader("📈 RR Over Time")
chart_placeholder = st.empty()

# Function to fetch latest data
def fetch_latest_data():
    response = supabase.table('maintable').select('*').order('timestamp', desc=True).limit(20).execute()
    return response.data if response.data else None

# Function to make predictions
def predict_category(stored_count_60s):
    if stored_count_60s == 0:
        return None  # No prediction if stored_count_60s is 0
    
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
        supabase.table("maintable").update({"prediction": prediction}).eq("id", record_id).execute()

# Keep track of last valid values
last_valid_stored_count = None
last_valid_prediction = None
last_valid_timestamp = None

# Main loop to process data in real-time
while True:
    try:
        latest_data_list = fetch_latest_data()
        
        if latest_data_list is None:
            status_placeholder.error("⚠️ Connect the device.")
            continue
        
        df = pd.DataFrame(latest_data_list)
        if df.empty:
            status_placeholder.error("⚠️ Connect the device.")
            continue
        
        df["count"] = df["count"].astype(int)
        df["count_60s"] = df["count_60s"].astype(int)
        df["stored_count_60s"] = df["stored_count_60s"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        latest_data = df.iloc[0]
        
        # Update the date and time placeholder with the latest timestamp
        latest_timestamp = latest_data["timestamp"].strftime("%A, %B %d, %Y | %H:%M:%S")
        datetime_placeholder.subheader(f"📅 {latest_timestamp}")

        # Check and update prediction
        if pd.isna(latest_data.get("prediction", None)):
            predicted_value = predict_category(latest_data["stored_count_60s"])
            update_supabase_prediction(latest_data["id"], predicted_value)
            latest_data["prediction"] = predicted_value
        
        # Store last valid values if there is a prediction
        if latest_data["prediction"] in ["Tachypnea", "Bradypnea", "Normal"]:
            last_valid_stored_count = latest_data["stored_count_60s"]
            last_valid_prediction = latest_data["prediction"]
            last_valid_timestamp = latest_data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        # Display patient chart
        data_table_placeholder.dataframe(df)

        # Display metrics
        live_count_placeholder.metric("📊 Live RR per minute", latest_data["count_60s"])
        total_count_placeholder.metric("📈 Total RR", latest_data["count"])

        # Display alert based on prediction
        if last_valid_prediction:
            if last_valid_prediction == "Normal":
                status_placeholder.success(f"✅ Normal ({last_valid_timestamp})\n📊 Stored Count: {last_valid_stored_count}")
            elif last_valid_prediction == "Tachypnea":
                status_placeholder.warning(f"⚠️ ALERT ({last_valid_timestamp}): Tachypnea detected!\n📊 Stored Count: {last_valid_stored_count}")
            elif last_valid_prediction == "Bradypnea":
                status_placeholder.error(f"🚨 CRITICAL ALERT ({last_valid_timestamp}): Bradypnea detected!\n📊 Stored Count: {last_valid_stored_count}")

        # Chart update
        fig = px.line(df, x="timestamp", y=["count_60s", "count"], 
                      title=f"Respiratory Rate Over Time (Latest: {latest_timestamp})",
                      labels={"timestamp": "Time", "count_60s": "RR per min", "count": "Total RR"})
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    
    except Exception as e:
        st.error(f"Error in main loop: {e}")
