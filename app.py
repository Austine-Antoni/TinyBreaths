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

model = load_model("LSTM_model.h5", compile = False)
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Respiratory Rate Dashboard", layout="wide")

# Create two columns
col1, col2 = st.columns([1.5, 5])  # Adjust width ratio as needed

# Insert image in the first column
with col1:
    st.image("TINY.png", width = 500)  # Adjust width as needed

# Insert title in the second column
with col2:
    st.markdown("<h1 style='margin-top: 70px; font-size: 60px;'>Respiratory Rate (RR) Monitoring Dashboard</h1>", unsafe_allow_html=True)
    # Placeholder for Date and Time
    datetime_placeholder = st.subheader("📅 Loading date and time...")

# Layout: Table on the left, Readings & Alerts on the right
col1, col2 = st.columns([5, 5])

with col1:
    st.subheader("📋 Patient Chart")
    data_table_placeholder = st.empty()

with col2:
    st.subheader("📊 RR")
    live_count_placeholder = st.empty()
    total_count_placeholder = st.empty()
    status_placeholder = st.empty()  # Status for normal/warnings

st.subheader("📈 Respiratory Rate Over Time")
chart_placeholder = st.empty()

# Function to fetch latest data
def fetch_latest_data():
    response = supabase.table('maintable').select('*').order('timestamp', desc=True).limit(20).execute()
    return response.data if response.data else []

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
last_data_timestamp = None  # Track the timestamp of the last received data

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
            last_data_timestamp = latest_data["timestamp"]
            
            # Update the date and time placeholder with the latest timestamp
            latest_timestamp = last_data_timestamp.strftime("%A, %B %d, %Y | %H:%M:%S")
            datetime_placeholder.subheader(f"📅 {latest_timestamp}")

            # Check and update prediction
            if pd.isna(latest_data.get("prediction", None)):
                predicted_value = predict_category(latest_data["stored_count_60s"])
                update_supabase_prediction(latest_data["id"], predicted_value)
                df.at[df.index[0], "prediction"] = predicted_value
            
            # Store last valid values if there is a prediction
            if latest_data["prediction"] in ["Tachypnea", "Bradypnea", "Normal"]:
                last_valid_stored_count = latest_data["stored_count_60s"]
                last_valid_prediction = latest_data["prediction"]
                last_valid_timestamp = last_data_timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Display patient chart
            data_table_placeholder.dataframe(df)

            # Display metrics
            live_count_placeholder.metric("📊 Live RR per minute", latest_data["count_60s"], border=True)
            total_count_placeholder.metric("📈 Total RR", latest_data["count"], border=True)

            # Display alert based on prediction
            if last_valid_prediction:
                if last_valid_prediction == "Normal":
                    status_placeholder.success(f"✅ Normal \n📊 Stored Count: {last_valid_stored_count} at ({last_valid_timestamp})")
                elif last_valid_prediction == "Tachypnea":
                    status_placeholder.warning(f"⚠️ ALERT: Tachypnea detected!\n📊 Stored Count: {last_valid_stored_count} at ({last_valid_timestamp})")
                elif last_valid_prediction == "Bradypnea":
                    status_placeholder.error(f"🚨 CRITICAL ALERT: Bradypnea detected!\n📊 Stored Count: {last_valid_stored_count} at ({last_valid_timestamp})")

            # Chart update
            fig = px.line(df, x="timestamp", y=["count_60s", "count"], 
                          title=f"RR Over Time (Latest: {latest_timestamp})",
                          labels={"timestamp": "Time", "count_60s": "RR per min", "count": "Total RR"})
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")

    
    except Exception as e:
        st.error(f"Error in main loop: {e}")
