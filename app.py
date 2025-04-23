import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# 💅 Custom CSS styling
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .stApp {
        background-color: #fefefe;
    }
    .stButton>button {
        background-color: #007acc;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stSelectbox, .stNumberInput {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 📦 Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# 🖼️ App header
st.image("https://cdn-icons-png.flaticon.com/512/1055/1055687.png", width=80)
st.markdown("<h1 style='text-align: center;'>Laptop Price Predictor 💻</h1>", unsafe_allow_html=True)
st.write("### 🔧 Fill the laptop specs below:")

# 🔲 Two-column layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('🖥️ Brand', ['Select...'] + list(df['Company'].unique()))
    laptop_type = st.selectbox('📘 Laptop Type', ['Select...'] + list(df['TypeName'].unique()))
    ram = st.selectbox('💾 RAM (in GB)', ['Select...'] + sorted(df['Ram'].unique()))
    touchscreen = st.selectbox('🖐️ Touchscreen', ['Select...', 'Yes', 'No'])
    screen_size = st.number_input('📏 Screen Size (inches)', min_value=10.0, max_value=18.0, step=0.1)
    hdd = st.selectbox('🗃️ HDD (GB)', ['Select...'] + list(df['HDD'].unique()))

with col2:
    weight = st.number_input('⚖️ Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
    ips = st.selectbox('📺 IPS Display', ['Select...', 'Yes', 'No'])
    resolution = st.selectbox('🖼️ Screen Resolution', ['Select...', '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800'])
    cpu = st.selectbox('🧠 CPU', ['Select...'] + list(df['Cpu brand'].unique()))
    ssd = st.selectbox('⚙️ SSD (GB)', ['Select...'] + list(df['SSD'].unique()))
    gpu = st.selectbox('🎮 GPU Brand', ['Select...'] + list(df['Gpu brand'].unique()))
    os = st.selectbox('💽 Operating System', ['Select...'] + list(df['os'].unique()))

# 📍 Input validation
def is_valid():
    return 'Select...' not in [company, laptop_type, ram, touchscreen, ips, resolution, cpu, hdd, ssd, gpu, os]

# 🔮 Predict button
st.markdown("---")
if st.button('🔮 Predict Laptop Price'):
    if not is_valid():
        st.warning("⚠️ Please make sure all dropdowns are selected.")
    else:
        try:
            # Convert categorical to binary
            touchscreen = 1 if touchscreen == 'Yes' else 0
            ips = 1 if ips == 'Yes' else 0

            # Calculate PPI
            x_res, y_res = map(int, resolution.split('x'))
            ppi = ((x_res**2 + y_res**2)**0.5) / screen_size

            # Create input DataFrame
            input_df = pd.DataFrame([[company, laptop_type, int(ram), weight, touchscreen, ips,
                                      ppi, cpu, int(hdd), int(ssd), gpu, os]],
                                    columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen',
                                             'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

            with st.spinner('🔄 Predicting...'):
                prediction = pipe.predict(input_df)[0]
                final_price = np.exp(prediction)

            # 📊 Show result
            st.metric(label="💰 Estimated Laptop Price", value=f"₹{round(final_price, 2):,}")

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")

# 📑 Sidebar
st.sidebar.title("📋 About This App")
st.sidebar.info("""
This is a machine learning-powered laptop price prediction tool.

Built using:
- Streamlit
- scikit-learn
- XGBoost
- Real market data

👨‍💻 Created by: Your Name
""")
