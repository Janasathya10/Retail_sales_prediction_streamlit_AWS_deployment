import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Dynamically get the root directory of the project
base_dir = os.path.dirname(os.path.dirname(__file__))  # one level up from /app

# Load the trained model and scaler using full paths
model = load_model(os.path.join(base_dir, "models", "ann_model.keras"))
scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))

st.title("Retail Sales Forecast App")

# Input features
store = st.number_input("Store", min_value=1)
dept = st.number_input("Department", min_value=1)
size = st.number_input("Store Size", min_value=10000)
temperature = st.slider("Temperature", 0.0, 120.0)
fuel_price = st.slider("Fuel Price", 1.0, 5.0)
cpi = st.slider("CPI", 100.0, 250.0)
unemployment = st.slider("Unemployment", 0.0, 15.0)
markdowns = [st.slider(f"MarkDown{i+1}", 0.0, 5000.0) for i in range(5)]

# Combine inputs
input_data = np.array([[store, dept, size, temperature, fuel_price, cpi, unemployment] + markdowns])

# Scale inputs
scaled_input = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_input)[0][0]
st.success(f"Predicted Weekly Sales: ${prediction:,.2f}")