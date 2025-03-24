import os
import datetime
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from supabase import create_client
import time
import math

# 🔹 Initialize Supabase connection
API_URL = 'https://ocrlmdadtekazfnhmquj.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jcmxtZGFkdGVrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1MTA2MzksImV4cCI6MjA1NzA4NjYzOX0.25bkWBV3v4cyjcA_-dUL8-IK3fSywARfVQ82UsZPelc'  
supabase = create_client(API_URL, API_KEY)

# 🔹 Load ML Model & Scaler
model = load_model("LSTM_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# 🔹 Create two columns for UI layout
col1, col2 = st.columns([1.5, 5])

# 🔹 Insert image in the first column
with col1:
    st.image("TINY.png")

# 🔹 Insert title in the second column
with col2:
    st.title("Respiratory Rate (RR) Monitoring Dashboard")
    datetime_placeholder = st.subheader("📅 Loading date and time...")

# 🔹 Create Layout: Table on the left, Readings & Alerts on the right
col1, col2 = st.columns([5, 2])

with col1:
    st.subheader("📋 Patient Chart")
    data_table_placeholder = st.empty()

with col2:
    st.subheader("📊 Respiratory Rate")
    live_count_placeholder = st.empty()
    total_count_placeholder = st.empty()
    status_placeholder = st.empty()  # Status indicator

st.subheader("📈 Respiratory Rate Over Time")
chart_placeholder = st.empty()

# 🔹 Function to fetch latest data from Supabase
def fetch_latest_data():
    response = supabase.table('maintable').select('*').order('timestamp', desc=True).limit(20).execute()
    return response.data if response.data else []

# 🔹 Function to predict category using the ML model
def predict_category(stored_count_60s):
    if stored_count_60s == 0:
        return None  # Ignore zero values
    
    new_input = np.array([[60, stored_count_60s]])
    new_input_scaled = scaler.transform(new_input.reshape(-1, 2))
    new_input_reshaped = new_input_scaled.reshape(1, 1, 2)

    new_prediction = model.predict(new_input_reshaped)
    predicted_category = np.argmax(new_prediction, axis=1)

    category_map = ['Bradypnea', 'Normal', 'Tachypnea']
    return category_map[int(predicted_category[0])]

# 🔹 Function to update Supabase with new prediction
def update_supabase_prediction(record_id, prediction):
    if prediction and not (isinstance(prediction, float) and math.isnan(prediction)):
        supabase.table("maintable").update({"prediction": prediction}).eq("id", record_id).execute()

# 🔹 Initialize session state for persistent display
if "last_valid_prediction" not in st.session_state:
    st.session_state.last_valid_prediction = None
    st.session_state.last_valid_stored_count = None
    st.session_state.last_valid_timestamp = None

# 🔹 Fetch latest data
latest_data_list = fetch_latest_data()

if latest_data_list:
    df = pd.DataFrame(latest_data_list)
    df["count"] = df["count"].astype(int)
    df["count_60s"] = df["count_60s"].astype(int)
    df["stored_count_60s"] = df["stored_count_60s"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    latest_data = df.iloc[0]
    latest_timestamp = latest_data["timestamp"].strftime("%A, %B %d, %Y | %H:%M:%S")
    datetime_placeholder.subheader(f"📅 {latest_timestamp}")

    stored_count = latest_data.get("stored_count_60s", None)

    # 🔹 Get Prediction
    current_prediction = predict_category(stored_count) if stored_count else None

    # 🔹 Update Supabase with the new prediction if missing
    if pd.isna(latest_data.get("prediction")) or latest_data.get("prediction") is None:
        if current_prediction:
            update_supabase_prediction(latest_data["id"], current_prediction)
            df.at[df.index[0], "prediction"] = current_prediction  # Update dataframe locally

    # 🔹 Update session state **only if** the prediction has changed
    if current_prediction in ["Tachypnea", "Bradypnea", "Normal"] and current_prediction != st.session_state.last_valid_prediction:
        st.session_state.last_valid_prediction = current_prediction
        st.session_state.last_valid_stored_count = stored_count
        st.session_state.last_valid_timestamp = latest_data["timestamp"]

    # 🔹 Display status message based on the last valid prediction
    if st.session_state.last_valid_prediction == "Normal":
        status_placeholder.success(
            f"✅ Normal \n📊 Stored Count: {st.session_state.last_valid_stored_count} at {st.session_state.last_valid_timestamp}"
        )
    elif st.session_state.last_valid_prediction == "Tachypnea":
        status_placeholder.warning(
            f"⚠️ ALERT: Tachypnea detected!\n📊 Stored Count: {st.session_state.last_valid_stored_count} at {st.session_state.last_valid_timestamp}"
        )
    elif st.session_state.last_valid_prediction == "Bradypnea":
        status_placeholder.error(
            f"🚨 CRITICAL ALERT: Bradypnea detected!\n📊 Stored Count: {st.session_state.last_valid_stored_count} at {st.session_state.last_valid_timestamp}"
        )
    
    # 🔹 Display metrics
    live_count_placeholder.metric("📊 Live RR per minute", latest_data["count_60s"])
    total_count_placeholder.metric("📈 Total RR", latest_data["count"])

    # 🔹 Update chart
    fig = px.line(df, x="timestamp", y=["count_60s", "count"], 
                  title=f"RR Over Time (Latest: {latest_timestamp})",
                  labels={"timestamp": "Time", "count_60s": "RR per min", "count": "Total RR"})
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")

# 🔹 Auto-refresh every 5 seconds to check for new data
st.rerun()
