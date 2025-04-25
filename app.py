import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.markdown("## Laptop Price Prediction")
st.write("Fill the details below to predict laptop price:")

st.sidebar.markdown("""
## About this App
This app predicts the price of a laptop based on several features such as brand, RAM, CPU, screen size, and more.
It uses a machine learning model that has been trained on historical laptop data.
Fill in the details and get an estimated price for your laptop!

## How to Use
1. Select the brand of the laptop.
2. Choose the laptop type (e.g., gaming, business).
3. Enter specifications such as RAM size, screen size, etc.
4. Hit the "Predict Price" button to get an estimated laptop price

## INSTRUCTION :
## CPU :
  ....Apple uses its own CPUs now — called M1, M2, M3 series.
#RAM : 
1. Please Select the Ram 4 or greater than 4 (2 gb ram is not available in market)
2. Apple laptop have Ram Greater than  and Equal to the 8 GB .\n\n
#Laptop Type :
1. Apple do not offer 2-in-1 Convertible and Netbook.
2. Dell do not offer Netbook.
3. Chuwi do not offer Gaming and Workstation.
4. MSI do not offer Netbook and 2-in-1 Convertible.
5. Microsoft do not offer Netbook and Workstation.
6. Toshiba do not offer Gaming .
7. Huawei do not offer Netbook, Gaming,Workstation.
8. Vero do not offer Netbook,Gaming,2-in-1 Convertible,Workstation.
9. Razer do not offer Netbook and 2-in-1 Convertible.
10. Mediacom do not offer Ultrabook, Netbook, Gaming, and Workstation.
11. Samsung do not offer Netbook and Workstation.
12. Google do not offer Netbook, Gaming, and Workstation.
13. Fujitsu do not offer Gaming and Workstation.
14. LG do not offer Netbook, Gaming, and Workstation.

## Model Description
This app uses a machine learning model built on a dataset of historical laptop prices. 
The model considers features like brand, CPU, RAM, SSD size, and display resolution to predict the price of the laptop.
The training process uses a **Random Forest Regressor** to estimate the price accurately.

## Disclaimer
The predicted price is an estimate and may not reflect the actual market price. Prices vary by region and seller.
This app is for educational purposes only.
""")

company = st.selectbox('Brand', ['Select...'] + list(df['Company'].unique()))
laptop_type = st.selectbox('Laptop Type', ['Select...'] + list(df['TypeName'].unique()))
ram = st.selectbox('RAM (in GB)', ['Select...'] + sorted(df['Ram'].unique()))
weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
touchscreen_option = st.selectbox('Touchscreen', ['Select...', 'Yes', 'No'])  # keep string version for logic
ips = st.selectbox('IPS Display', ['Select...', 'Yes', 'No'])
screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=18.0, step=0.1)
resolution = st.selectbox('Screen Resolution', ['Select...', '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800'])

cpu = st.selectbox('CPU', ['Select...'] + list(df['Cpu brand'].unique()))
hdd = st.selectbox('HDD (GB)', ['Select...'] + list(df['HDD'].unique()))
ssd = st.selectbox('SSD (GB)', ['Select...'] + list(df['SSD'].unique()))
gpu = st.selectbox('GPU Brand', ['Select...'] + list(df['Gpu brand'].unique()))
os = st.selectbox('Operating System', ['Select...'] + list(df['os'].unique()))

# Convert to binary
touchscreen = 1 if touchscreen_option == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Calculate PPI
x_res, y_res = map(int, resolution.split('x')) if resolution != 'Select...' else (0, 0)
ppi = ((x_res**2 + y_res**2)**0.5) / screen_size if resolution != 'Select...' else 0

# Predict button
if st.button('## Predict Price'):
    # Check if all dropdowns have been selected
    if 'Select...' in [company, laptop_type, ram, touchscreen_option, ips, resolution, cpu, hdd, ssd, gpu, os]:
        st.warning(" Please make sure all dropdowns are selected.")
    else:
        # Prepare input as DataFrame
        input_df = pd.DataFrame([[company, laptop_type, int(ram), weight, touchscreen, ips,
                                  ppi, cpu, hdd, ssd, gpu, os]],
                                columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen',
                                         'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

        # Predict
        try:
            prediction = pipe.predict(input_df)[0]
            final_price = np.exp(prediction)  # Reverse log transform

            # Manual adjustment if touchscreen = Yes
            if touchscreen == 1:
                final_price += 9000
                

            st.success(f"## Estimated Laptop Price: ₹{round(final_price, 2):,}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
