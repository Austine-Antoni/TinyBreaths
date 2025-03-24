import os
import datetime
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from supabase import create_client

global last_valid_prediction  # Store the last valid category

# Supabase connection
API_URL = 'https://ocrlmdadtekazfnhmquj.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jcmxtZGFkdGVrYXpmbmhtcXVqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1MTA2MzksImV4cCI6MjA1NzA4NjYzOX0.25bkWBV3v4cyjcA_-dUL8-IK3fSywARfVQ82UsZPelc'  
supabase = create_client(API_URL, API_KEY)
    
model = load_model("LSTM_model.h5", compile = False)
scaler = joblib.load("scaler.pkl")

# Create two columns
col1, col2 = st.columns([1.5, 5])  # Adjust width ratio as needed

# Insert image in the first column
with col1:
    st.image("TINY.png")  # Adjust width as needed

# Insert title in the second column
with col2:
    st.title("Respiratory Rate (RR) Monitoring Dashboard")
    # Placeholder for Date and Time
    datetime_placeholder = st.subheader("📅 Loading date and time...")

# Layout: Table on the left, Readings & Alerts on the right
col1, col2 = st.columns([5, 2])

with col1:
    st.subheader("📋 Patient Chart")
    data_table_placeholder = st.empty()

with col2:
    st.subheader("📊 Respiratory Rate")
    live_count_placeholder = st.empty()
    total_count_placeholder = st.empty()
    status_placeholder = st.empty()  # Status for normal/warnings

st.subheader("📈 Respiratory Rate Over Time")
chart_placeholder = st.empty()

# **Functions**
def fetch_latest_data():
    response = supabase.table('maintable').select('*').order('timestamp', desc=True).limit(20).execute()
    return response.data if response.data else []

def predict_category(stored_count_60s):
    if stored_count_60s == 0:
        return None  
    new_input = np.array([[60, stored_count_60s]])
    new_input_scaled = scaler.transform(new_input.reshape(-1, 2))
    new_input_reshaped = new_input_scaled.reshape(1, 1, 2)
    
    new_prediction = model.predict(new_input_reshaped)
    predicted_category = np.argmax(new_prediction, axis=1)

    category_map = ['Bradypnea', 'Normal', 'Tachypnea']
    return category_map[int(predicted_category[0])]

def update_supabase_prediction(record_id, prediction):
    if prediction and not (isinstance(prediction, float) and math.isnan(prediction)):
        supabase.table("maintable").update({"prediction": prediction}).eq("id", record_id).execute()

# **Tracking last valid values**
last_known_timestamp = None
last_known_prediction = None

# **Main loop**
while True:
    try:
        latest_data_list = fetch_latest_data()
        
        if latest_data_list:
            df = pd.DataFrame(latest_data_list)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            latest_data = df.iloc[0]
            new_timestamp = latest_data["timestamp"]
            stored_count = latest_data.get("stored_count_60s", None)

            # **Only process if data is new**
            if new_timestamp != last_known_timestamp:
                last_known_timestamp = new_timestamp  # Update timestamp tracking

                # **Update UI time**
                latest_timestamp_str = new_timestamp.strftime("%A, %B %d, %Y | %H:%M:%S")
                datetime_placeholder.subheader(f"📅 {latest_timestamp_str}")

                # **Predict category**
                if stored_count and stored_count > 0:
                    current_prediction = predict_category(stored_count)
                else:
                    current_prediction = None

                # **Update Supabase if needed**
                if pd.isna(latest_data.get("prediction")) or latest_data.get("prediction") is None:
                    if current_prediction:
                        update_supabase_prediction(latest_data["id"], current_prediction)
                        df.at[df.index[0], "prediction"] = current_prediction  

                # **Only update UI if prediction changes**
                if current_prediction and current_prediction != last_known_prediction:
                    last_known_prediction = current_prediction

                    # **Update patient chart**
                    data_table_placeholder.dataframe(df)

                    # **Update metrics**
                    live_count_placeholder.metric("📊 Live RR per minute", latest_data["count_60s"])
                    total_count_placeholder.metric("📈 Total RR", latest_data["count"])

                    # **Update chart**
                    fig = px.line(df, x="timestamp", y=["count_60s", "count"], 
                                  title=f"RR Over Time (Latest: {latest_timestamp_str})",
                                  labels={"timestamp": "Time", "count_60s": "RR per min", "count": "Total RR"})
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # **Update status display**
                    status_placeholder.empty()
                    if current_prediction == "Normal":
                        status_placeholder.success(f"✅ Normal \n📊 Stored Count: {stored_count} at {latest_timestamp_str}")
                    elif current_prediction == "Tachypnea":
                        status_placeholder.warning(f"⚠️ ALERT: Tachypnea detected!\n📊 Stored Count: {stored_count} at {latest_timestamp_str}")
                    elif current_prediction == "Bradypnea":
                        status_placeholder.error(f"🚨 CRITICAL ALERT: Bradypnea detected!\n📊 Stored Count: {stored_count} at {latest_timestamp_str}")

    except Exception as e:
        st.error(f"Error in main loop: {e}")
