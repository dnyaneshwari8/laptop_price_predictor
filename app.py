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
    ppi = 0  # If resolution is not selected, set ppi to a default value (0)

# Handle invalid HDD and SSD values (make sure they're integers)
hdd = int(hdd) if hdd != 'Select...' else 0
ssd = int(ssd) if ssd != 'Select...' else 0

# Predict button
if st.button('🔮 Predict Price'):
    if 'Select...' in [company, laptop_type, ram, touchscreen, ips, resolution, cpu, hdd, ssd, gpu, os]:
        st.warning("⚠️ Please make sure all dropdowns are selected.")
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
            st.success(f"💰 Estimated Laptop Price: ₹{round(final_price, 2):,}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
