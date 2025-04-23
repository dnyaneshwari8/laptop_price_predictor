import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


st.markdown("## 💻 Laptop Price Prediction")
st.write("Fill the details below to predict laptop price:")

st.sidebar.markdown(" laptop price prediction ")

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
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Calculate PPI
x_res, y_res = map(int, resolution.split('x')) if resolution != 'Select...' else (0, 0)
ppi = ((x_res**2 + y_res**2)**0.5) / screen_size if resolution != 'Select...' else 0

# Predict button
if st.button('🔮 Predict Price'):
    # Check if all dropdowns have been selected
    if 'Select...' in [company, laptop_type, ram, touchscreen, ips, resolution, cpu, hdd, ssd, gpu, os]:
        st.warning(" Please make sure all dropdowns are selected.")
    else:
        # Prepare input as DataFrame (important for pipeline compatibility)
        input_df = pd.DataFrame([[company, laptop_type, int(ram), weight, touchscreen, ips,
                                  ppi, cpu, hdd, ssd, gpu, os]],
                                columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen',
                                         'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

        # Predict
        try:
            prediction = pipe.predict(input_df)[0]
            final_price = np.exp(prediction)  # Reverse log transform
            st.success(f" ....Estimated Laptop Price: ₹{round(final_price, 2):,}")
        except Exception as e:
            st.error(f" Prediction Error: {e}")
