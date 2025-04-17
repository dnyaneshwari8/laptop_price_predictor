import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set Streamlit page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Load the trained pipeline and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title
st.markdown("## 💻 Laptop Price Prediction")
st.write("Fill the details below to predict laptop price:")

# --- Critical Fix 1: Ensure consistent brand order with training data ---
# Get brands in the EXACT SAME ORDER as during model training
brands_sorted = sorted(df['Company'].unique())  # Alphabetical order or match training data

# --- Critical Fix 2: Input validation for Apple specs ---
def validate_apple_specs(company, laptop_type, os):
    if company == 'Apple':
        if laptop_type != 'Ultrabook':
            st.warning("⚠️ Apple laptops are typically Ultrabooks")
        if os != 'Mac':
            st.warning("⚠️ Apple laptops usually run macOS")

# Input fields
company = st.selectbox('Brand', brands_sorted)  # Use sorted brands
laptop_type = st.selectbox('Laptop Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
ips = st.selectbox('IPS Display', ['Yes', 'No'])
screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=18.0, step=0.1)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800'])

cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', df['HDD'].unique())
ssd = st.selectbox('SSD (GB)', df['SSD'].unique())
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# --- Critical Fix 3: Apple-specific validation ---
validate_apple_specs(company, laptop_type, os)

# Preprocessing
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Calculate PPI
x_res, y_res = map(int, resolution.split('x'))
ppi = ((x_res**2 + y_res**2)**0.5) / screen_size

# Predict button
if st.button('🔮 Predict Price'):
    # --- Critical Fix 4: Verify Apple specs consistency ---
    if company == 'Apple' and (os != 'Mac' or laptop_type not in ['Ultrabook', 'Notebook']):
        st.error("❌ Invalid Apple configuration! Check OS and laptop type.")
    else:
        input_df = pd.DataFrame([[company, laptop_type, ram, weight, touchscreen, ips,
                                ppi, cpu, hdd, ssd, gpu, os]],
                                columns=df.columns.drop('Price'))  # Match training columns exactly
        
        try:
            prediction = pipe.predict(input_df)[0]
            final_price = np.exp(prediction)  # Reverse log transform
            
            # --- Critical Fix 5: Price sanity check ---
            min_price = df[df['Company'] == company]['Price'].min()
            max_price = df[df['Company'] == company]['Price'].max()
            
            if not (min_price <= final_price <= max_price):
                st.warning(f"⚠️ Unusual prediction! Typical {company} prices: ₹{min_price:,} - ₹{max_price:,}")
            
            st.success(f"💰 Estimated Laptop Price: ₹{round(final_price, 2):,}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
