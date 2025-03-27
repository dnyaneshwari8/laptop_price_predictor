import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set custom page config
st.set_page_config(page_title="Laptop Price Predictor 💻", page_icon="💻", layout="centered")

# Custom CSS styling (adapted from your CSS)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background: url("https://th.bing.com/th/id/OIP.Fwm2-Eis29GeC_ix61m6LAHaFt?w=579&h=446&rs=1&pid=ImgDetMain") no-repeat center center fixed;
            background-size: cover;
        }

        .stApp {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 15px;
            max-width: 600px;
            margin: auto;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        h1, h2, h3, h4 {
            color: #333;
            text-align: center;
        }

        label, .stSelectbox label, .stNumberInput label {
            font-weight: bold;
            color: #444;
        }

        .stButton>button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #218838;
        }

        .stMarkdown h3 {
            font-size: 1.2rem;
            color: #155724;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title
st.markdown("## 💻 Laptop Price Prediction App")
st.write("Fill the details below to predict laptop price:")

# Input fields
company = st.selectbox('Brand', df['Company'].unique())
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

# Preprocessing
is_touchscreen = 1 if touchscreen == 'Yes' else 0
is_ips = 1 if ips == 'Yes' else 0

# Calculate PPI
x_res, y_res = map(int, resolution.split('x'))
ppi = ((x_res**2 + y_res**2)**0.5) / screen_size

# Predict button
if st.button('🔮 Predict Price'):
    input_df = pd.DataFrame([[company, laptop_type, ram, weight, is_touchscreen, is_ips,
                              ppi, cpu, hdd, ssd, gpu, os]],
                            columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen',
                                     'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])
    try:
        prediction = pipe.predict(input_df)[0]
        final_price = np.exp(prediction)  # Reverse log transform
        st.markdown(f"### 💰 Estimated Laptop Price: ₹{round(final_price, 2):,}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
