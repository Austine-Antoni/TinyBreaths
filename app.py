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

st.set_page_config(page_title="Tiny Breaths", page_icon="TINY.png", layout="wide")
# Create two columns
col1, col2 = st.columns([1, 5])  # Adjust width ratio as needed

# Insert image in the first column
with col1:
    st.image("TINY.png", width = 200)  # Adjust width as needed

# Insert title in the second column
with col2:
    st.title("Respiratory Rate (RR) Monitoring Dashboard")
    # Placeholder for Date and Time
    datetime_placeholder = st.subheader("📅 Loading date and time...")

# Layout: Table on the left, Readings & Alerts on the right
col1, col2 = st.columns([5, 5])

with col1:
    st.subheader("📋 Patient Chart")
    data_table_placeholder = st.empty()

with col2:
    st.subheader("📊 Respiratory Rate")
    live_count_placeholder = st.empty()
    total_count_placeholder = st.empty()
    status_placeholder = st.empty()  # Status for normal/warnings
    warning_placeholder = st.empty()  

st.subheader("📈 Respiratory Rate Over Time")
chart_placeholder = st.empty()

# Function to fetch latest data
def fetch_latest_data():
    response = supabase.table('maintable').select('*').order('timestamp', desc=True).limit(50).execute()
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
    if prediction is not None and not (isinstance(prediction, float) and math.isnan(prediction)):
        supabase.table("maintable").update({"diagnosis": prediction}).eq("id", record_id).execute()

# def play_sound():
#     """Inject JavaScript to play a sound in the browser"""
#     audio_url = "alert.wav"  # Update with your actual file URL
#     sound_script = f"""
#     <script>
#     var audio = new Audio("{audio_url}");
#     audio.play();
#     </script>
#     """
#     st.markdown(sound_script, unsafe_allow_html=True)
    
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
            current_count_60s = latest_data["count_60s"]
            current_timestamp = latest_data["timestamp"]
            
            # Update the date and time placeholder with the latest timestamp
            latest_timestamp = last_data_timestamp.strftime("%A, %B %d, %Y | %H:%M:%S")
            datetime_placeholder.subheader(f"📅 {latest_timestamp}")

            # Predict category directly in the code
            stored_count = latest_data.get("stored_count_60s", None)

            if stored_count is not None and stored_count > 0:
                current_prediction = predict_category(stored_count)
            else:
                current_prediction = None

            # Update Supabase only if no prediction exists in the database
            if pd.isna(latest_data.get("diagnosis")) or latest_data.get("diagnosis") is None:
                if current_prediction:
                    update_supabase_prediction(latest_data["id"], current_prediction)
                    df.at[df.index[0], "diagnosis"] = current_prediction  # Update dataframe locally
                    
            # Store last valid values
            if current_prediction in ["Tachypnea", "Bradypnea", "Normal"]:
                last_valid_stored_count = stored_count
                last_valid_prediction = current_prediction
                last_valid_timestamp = last_data_timestamp  # Update timestamp

            # Display patient chart
            data_table_placeholder.dataframe(df)


            # Clears previous status
            status_placeholder.empty()  
            
            if last_valid_prediction:
                # Display status based on the last valid prediction
                if last_valid_prediction == "Normal":
                    status_placeholder.success(
                        f"✅ Normal \n📊 Stored Count: {last_valid_stored_count} at {last_valid_timestamp}"
                    )
                elif last_valid_prediction == "Tachypnea":
                    status_placeholder.warning(
                        f"⚠️ ALERT: Tachypnea detected!\n📊 Stored Count: {last_valid_stored_count} at {last_valid_timestamp}"
                    )
                    # Play sound in a background thread (only for Tachypnea and Bradypnea)
                    # play_sound() 
                elif last_valid_prediction == "Bradypnea":
                    status_placeholder.error(
                        f"🚨 CRITICAL ALERT: Bradypnea detected!\n📊 Stored Count: {last_valid_stored_count} at {last_valid_timestamp}"
                    )
                    # Play sound in a background thread (only for Tachypnea and Bradypnea)
                    # play_sound() 

            # Check if count_60s has remained unchanged for 10 seconds
            if "last_count_60s" not in st.session_state:
                st.session_state["last_count_60s"] = current_count_60s
                st.session_state["last_count_time"] = datetime.datetime.now()
            
            elapsed_time = (datetime.datetime.now() - st.session_state["last_count_time"]).total_seconds()
            
            if current_count_60s != st.session_state["last_count_60s"]:
                # Reset timer if count_60s changes
                st.session_state["last_count_60s"] = current_count_60s
                st.session_state["last_count_time"] = datetime.datetime.now()
                warning_placeholder.empty()  # Clear warning
            
            elif elapsed_time >= 10:
                # Show warning if no change in 10 seconds
                warning_placeholder.warning("⚠️ WARNING: No detected movement for 10 seconds!")
                
            # Display metrics
            live_count_placeholder.metric("📊 Live RR per minute", latest_data["count_60s"])
            total_count_placeholder.metric("📈 Total RR", latest_data["count"])
            
            # Chart update
            fig = px.line(df, x="timestamp", y=["count_60s"], 
                          title=f"RR Over Time (Latest: {latest_timestamp})",
                          labels={"timestamp": "Time", "count_60s": "RR per min"})
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")

    
    except Exception as e:
        st.error(f"Error in main loop: {e}")
