import os
import time
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

# Load ML Model & Scaler
model = load_model("LSTM_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# App Title
st.title("Real-Time Respiratory Rate (RR) Monitoring")

# Placeholders
datetime_placeholder = st.subheader("📅 Loading date and time...")
col1, col2 = st.columns([2, 1.5])
with col1:
    data_table_placeholder = st.empty()
with col2:
    live_count_placeholder = st.empty()
    total_count_placeholder = st.empty()
    status_placeholder = st.empty()
chart_placeholder = st.empty()

# Store last timestamp to fetch only new data
if "last_data_timestamp" not in st.session_state:
    st.session_state.last_data_timestamp = None


# ⏳ Fetch only new data from Supabase
def fetch_latest_data():
    query = supabase.table("maintable").select("id, timestamp, count, count_60s, stored_count_60s, prediction").order("timestamp", desc=True).limit(20)
    
    # Fetch only new records if a previous timestamp exists
    if st.session_state.last_data_timestamp:
        query = query.gt("timestamp", st.session_state.last_data_timestamp)

    response = query.execute()
    return response.data if response.data else []


# 🔮 Predict category (Optimize: Cache Predictions)
def predict_category(stored_count_60s):
    if stored_count_60s == 0:
        return None

    # Avoid redundant predictions
    if stored_count_60s in st.session_state:
        return st.session_state[stored_count_60s]

    # Prepare input
    new_input = np.array([[60, stored_count_60s]], dtype=np.float32)
    new_input_scaled = scaler.transform(new_input)
    new_input_reshaped = new_input_scaled.reshape(1, 1, 2)

    # Predict
    new_prediction = model.predict(new_input_reshaped, verbose=0)
    category_map = ["Bradypnea", "Normal", "Tachypnea"]
    result = category_map[np.argmax(new_prediction)]

    # Cache result
    st.session_state[stored_count_60s] = result
    return result


# 🔄 Update Supabase only if needed
def update_supabase_prediction(record_id, prediction):
    if prediction:
        supabase.table("maintable").update({"prediction": prediction}).eq("id", record_id).execute()


# 📌 Main update function
def update_dashboard():
    latest_data_list = fetch_latest_data()
    if not latest_data_list:
        return

    df = pd.DataFrame(latest_data_list)
    df["count"] = df["count"].astype(int)
    df["count_60s"] = df["count_60s"].astype(int)
    df["stored_count_60s"] = df["stored_count_60s"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Get latest row
    latest_data = df.iloc[0]
    st.session_state.last_data_timestamp = latest_data["timestamp"]

    # 🕒 Update date & time
    latest_timestamp = latest_data["timestamp"].strftime("%A, %B %d, %Y | %H:%M:%S")
    datetime_placeholder.subheader(f"📅 {latest_timestamp}")

    # 🔍 Predict & update Supabase if missing
    if pd.isna(latest_data.get("prediction", None)):
        predicted_value = predict_category(latest_data["stored_count_60s"])
        update_supabase_prediction(latest_data["id"], predicted_value)
        df.at[df.index[0], "prediction"] = predicted_value

    # 📊 Display Patient Chart
    data_table_placeholder.dataframe(df)

    # 📈 Display Metrics
    live_count_placeholder.metric("📊 Live RR per minute", latest_data["count_60s"])
    total_count_placeholder.metric("📈 Total RR", latest_data["count"])

    # 🚨 Alert Based on Prediction
    prediction = latest_data.get("prediction", "Unknown")
    stored_count = latest_data["stored_count_60s"]
    timestamp_str = latest_data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    if prediction == "Normal":
        status_placeholder.success(f"✅ Normal ({timestamp_str})\n📊 Stored Count: {stored_count}")
    elif prediction == "Tachypnea":
        status_placeholder.warning(f"⚠️ ALERT ({timestamp_str}): Tachypnea detected!\n📊 Stored Count: {stored_count}")
    elif prediction == "Bradypnea":
        status_placeholder.error(f"🚨 CRITICAL ALERT ({timestamp_str}): Bradypnea detected!\n📊 Stored Count: {stored_count}")

    # 📉 Update Chart (Every 5s)
    fig = px.line(df, x="timestamp", y=["count_60s", "count"], 
                  title=f"Respiratory Rate Over Time (Latest: {latest_timestamp})",
                  labels={"timestamp": "Time", "count_60s": "RR per min", "count": "Total RR"})
    chart_placeholder.plotly_chart(fig, use_container_width=True)


# ⏰ Auto Refresh Using Background Thread
def auto_refresh():
    while True:
        update_dashboard()
        time.sleep(2)  # Refresh every 2 seconds
        st.rerun()  # Force UI refresh

import threading
threading.Thread(target=auto_refresh, daemon=True).start()
