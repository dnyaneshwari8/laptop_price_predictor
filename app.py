import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set Streamlit page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Load the trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title
st.markdown("## 💻 Laptop Price Prediction")
st.write("Fill the details below to predict laptop price:")

# Dropdown and input fields
company = st.selectbox('Brand', ['Select...'] + list(df['Company'].unique()))
laptop_type = st.selectbox('Laptop Type', ['Select...'] + list(df['TypeName'].unique()))
ram = st.selectbox('RAM (in GB)', ['Select...'] + sorted(df['Ram'].unique()))
weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['Select...', 'Yes', 'No'])
ips = st.selectbox('IPS Display', ['Select...', 'Yes', 'No'])
screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=18.0, step=0.1)
resolution = st.selectbox('Screen Resolution', ['Select...', '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800'])
cpu = st.selectbox('CPU', ['Select...'] + list(df['Cpu brand'].unique()))
hdd = st.selectbox('HDD (GB)', ['Select...'] + list(df['HDD'].unique()))
ssd = st.selectbox('SSD (GB)', ['Select...'] + list(df['SSD'].unique()))
gpu = st.selectbox('GPU Brand', ['Select...'] + list(df['Gpu brand'].unique()))
os = st.selectbox('Operating System', ['Select...'] + list(df['os'].unique()))

# Preprocessing
if touchscreen == 'Yes':
    touchscreen = 1
elif touchscreen == 'No':
    touchscreen = 0

if ips == 'Yes':
    ips = 1
elif ips == 'No':
    ips = 0

# Validate before processing resolution
if resolution != 'Select...':
    x_res, y_res = map(int, resolution.split('x'))
    ppi = ((x_res**2 + y_res**2)**0.5) / screen_size
else:
    ppi = 0

# Predict button
if st.button('🔮 Predict Price'):
    if 'Select...' in [company, laptop_type, ram, touchscreen, ips, resolution, cpu, hdd, ssd, gpu, os]:
        st.warning("⚠️ Please make sure all dropdowns are selected.")
    else:
        # Prepare input as DataFrame (important for pipeline compatibility)
        input_df = pd.DataFrame([[company, laptop_type, int(ram), weight, touchscreen, ips,
                                  ppi, cpu, int(hdd), int(ssd), gpu, os]],
                                columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen',
                                         'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

        # Predict
        try:
            prediction = pipe.predict(input_df)[0]
            final_price = np.exp(prediction)  # Reverse log transform
            st.success(f"💰 Estimated Laptop Price: ₹{round(final_price, 2):,}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
