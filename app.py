import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# Page Configuration
st.set_page_config(
    page_title="Used Vehicle Price Prediction",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# 1. LOAD ASSETS (Cached for Performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        # Load the Full Pipeline (includes Preprocessing + XGBoost)
        pipeline = joblib.load('vehicle_price_pipeline.pkl')
        # Load the list of columns the model expects
        columns = joblib.load('input_columns.pkl')
        return pipeline, columns
    except FileNotFoundError:
        return None, None

pipeline, model_columns = load_assets()

# ------------------------------------------------------------------------------
# 2. UI LAYOUT
# ------------------------------------------------------------------------------
st.title("🚗 Used Vehicle Price Prediction")
st.markdown("### AI-Powered Valuation System")
st.markdown("---")

if pipeline is None:
    st.error("⚠️ Error: Model files not found. Please upload 'vehicle_price_pipeline.pkl' and 'input_columns.pkl'.")
    st.stop()

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("Vehicle Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        make = st.selectbox("Brand", ['Toyota', 'Honda', 'BMW', 'Maruti', 'Hyundai', 'Volkswagen', 'Lexus', 'Ford', 'Chevrolet', 'Nissan'])
        model_name = st.text_input("Model Name", value="Camry") # Text input handles high cardinality via TargetEncoder
        year = st.slider("Manufacturing Year", 2000, 2025, 2018)
        mileage = st.number_input("Odometer (Miles)", min_value=0, max_value=300000, value=45000, step=500)
        engine_hp = st.number_input("Horsepower (HP)", min_value=50, max_value=1000, value=180)
        transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'CVT'])
        
    with col2:
        fuel_type = st.selectbox("Fuel Type", ['Gasoline', 'Diesel', 'Electric', 'Hybrid'])
        drivetrain = st.selectbox("Drivetrain", ['FWD', 'RWD', 'AWD', '4WD'])
        condition = st.selectbox("Condition", ['Excellent', 'Good', 'Fair'])
        accident_history = st.selectbox("Accident History", ['None', 'Minor', 'Major'])
        seller_type = st.selectbox("Seller Type", ['Dealer', 'Private'])
        trim = st.text_input("Trim Level", value="SE")

    # Hidden defaults for required columns (Visual filler for simple UI)
    # In a full v2.0 app, you might fetch these from an API based on VIN
    exterior_color = "Black"
    interior_color = "Black"
    owner_count = 1
    brand_popularity = 0.5 
    body_type = "Sedan" 

    submit_btn = st.form_submit_button("💰 Calculate Value", type="primary")

# ------------------------------------------------------------------------------
# 3. PREDICTION LOGIC
# ------------------------------------------------------------------------------
if submit_btn:
    # A. Create DataFrame with User Inputs
    input_data = pd.DataFrame({
        'make': [make],
        'model': [model_name],
        'year': [year],
        'mileage': [mileage],
        'engine_hp': [engine_hp],
        'transmission': [transmission],
        'fuel_type': [fuel_type],
        'drivetrain': [drivetrain],
        'condition': [condition],
        'accident_history': [accident_history],
        'seller_type': [seller_type],
        'trim': [trim],
        # Defaults
        'body_type': [body_type],
        'exterior_color': [exterior_color],
        'interior_color': [interior_color],
        'owner_count': [owner_count],
        'brand_popularity': [brand_popularity]
    })

    # B. Apply EXACT Feature Engineering from Training (The "Mirror" Principle)
    # We use dynamic year calculation to ensure the app stays relevant
    current_year = datetime.datetime.now().year
    input_data['vehicle_age'] = current_year - input_data['year']
    
    # Avoid division by zero for new cars
    input_data['mileage_per_year'] = input_data['mileage'] / input_data['vehicle_age'].replace(0, 1)
    
    # Impute missing categorical values (Safety check)
    input_data['accident_history'] = input_data['accident_history'].fillna('None')
    
    # C. Prediction Execution
    try:
        # Predict using the Pipeline (Encoding -> Scaling -> XGBoost)
        # The pipeline handles all One-Hot and Target Encoding internally
        prediction = pipeline.predict(input_data)[0]
        
        # Display Result
        st.balloons()
        st.success(f"### 💵 Estimated Price: ${prediction:,.2f}")
        
        # Insights
        if accident_history != 'None':
            st.warning(f"⚠️ Note: Price impacted by '{accident_history}' accident history.")
            
        with st.expander("See Valuation Factors"):
            st.write(f"**Vehicle Age:** {input_data['vehicle_age'][0]} years")
            st.write(f"**Usage Intensity:** {int(input_data['mileage_per_year'][0]):,} miles/year")
            st.write(f"**Base Logic:** Depreciation curve applied to {make} {model_name}.")
            
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.info("Please ensure input format matches model requirements.")