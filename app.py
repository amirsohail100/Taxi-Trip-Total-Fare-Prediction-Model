import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import load
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Taxi Fare Predictor", layout="centered")

st.title("🚖 Taxi Fare Prediction")
st.write("Predict the total fare amount based on taxi trip details")

# Load pre-trained model and preprocessors
@st.cache_resource
def load_model():
    """Load pre-trained KNN model and preprocessors from pkl files"""
    model = load("models/KNN_model.pkl")
    scaler = load("models/scaler.pkl")
    columns = load("models/columns.pkl")
    
    # Create a label encoder for taxi colors
    lab_encod = LabelEncoder()
    lab_encod.classes_ = np.array(['yellow', 'green'])  # Adjust based on your data
    
    return model, scaler, lab_encod

# Load model and preprocessors
model, scaler, label_encoder = load_model()

# Create input columns
col1, col2 = st.columns(2)

with col1:
    passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)
    distance = st.number_input("Distance (miles)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
    fare = st.number_input("Fare ($)", min_value=0.0, max_value=500.0, value=15.0, step=0.5)

with col2:
    tip = st.number_input("Tip ($)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
    tolls = st.number_input("Tolls ($)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    color = st.selectbox("Taxi Color", label_encoder.classes_)

# Make prediction
if st.button("Predict Fare", type="primary"):
    # Encode color
    color_encoded = label_encoder.transform([color])[0]
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'passengers': [passengers],
        'distance': [distance],
        'fare': [fare],
        'tip': [tip],
        'tolls': [tolls],
        'color': [color_encoded]
    })
    
    # Scale the input
    col_to_scale = ['passengers', 'distance', 'fare', 'tip', 'tolls']
    input_scaled = input_data.copy()
    input_scaled[col_to_scale] = scaler.transform(input_data[col_to_scale])
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display results
    st.success("✅ Prediction Complete!")
    
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Predicted Total", f"${prediction:.2f}")
    with result_col2:
        manual_total = fare + tip + tolls
        st.metric("Manual Sum (Fare+Tip+Tolls)", f"${manual_total:.2f}")
    
    # Show trip summary
    st.write("---")
    st.subheader("Trip Summary")
    summary_data = {
        "Passengers": passengers,
        "Distance": f"{distance} miles",
        "Base Fare": f"${fare:.2f}",
        "Tip": f"${tip:.2f}",
        "Tolls": f"${tolls:.2f}",
        "Taxi Color": color,
        "Predicted Total": f"${prediction:.2f}"
    }
    
    for key, value in summary_data.items():
        st.write(f"**{key}:** {value}")
