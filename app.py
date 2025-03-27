import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and data safely
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
except FileNotFoundError:
    st.error("Required files 'pipe.pkl' or 'df.pkl' are missing. Please upload them to your project folder.")
    st.stop()

# Page title
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

# App heading
st.title("💻 Laptop Price Prediction ")
st.markdown("Fill the details below to predict laptop price:")

# Input form
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Laptop Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
ips = st.selectbox('IPS Display', ['Yes', 'No'])
screen_size = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=20.0, step=0.1)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (GB)', [0, 128, 256, 512, 1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):
    # Convert inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Extract resolution
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # Create input array
    query = np.array([[company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]])
    
    # Predict and show result
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.success(f"💰 Estimated Laptop Price: ₹{predicted_price}")
