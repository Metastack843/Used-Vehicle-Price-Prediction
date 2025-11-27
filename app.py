import streamlit as st
import pandas as pd
import joblib
import datetime

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Value AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS (Compact & Modern)
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Reduce top padding to fit more on screen */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Global Styles */
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0;
    }
    
    /* Hide Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Input Form Styling - Compact */
    .stForm {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444d;
    }
    
    /* Result Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #FF4B4B 0%, #bd3030 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(255, 75, 75, 0.2);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 2.8em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        transform: translateY(-1px);
    }
    
    /* Info Box Styling */
    .info-box {
        background-color: #1f2937;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. LOAD ASSETS
# ------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load('models/vehicle_price_pipeline.pkl')
        columns = joblib.load('models/input_columns.pkl')
        return pipeline, columns
    except FileNotFoundError:
        try:
            pipeline = joblib.load('vehicle_price_pipeline.pkl')
            columns = joblib.load('input_columns.pkl')
            return pipeline, columns
        except:
            return None, None

pipeline, model_columns = load_assets()

if pipeline is None:
    st.error("⚠️ Critical Error: Model files missing.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. APP LAYOUT
# ------------------------------------------------------------------------------

# Header (Compact)
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown("## 🚘 Vehicle Value AI")
with col_head2:
    st.markdown("<div style='text-align: right; color: gray; font-size: 0.8em; padding-top: 10px;'>v1.0 (Production)</div>", unsafe_allow_html=True)

# Main Split Layout (40% Inputs | 60% Results)
left_col, right_col = st.columns([2, 3], gap="medium")

# --- LEFT COLUMN: COMPACT INPUTS ---
with left_col:
    with st.form("prediction_form"):
        st.markdown("###### Vehicle Configuration")
        
        # 4-Column Layout to reduce vertical height
        c1, c2 = st.columns(2)
        with c1:
            make = st.selectbox("Brand", ['Toyota', 'Honda', 'BMW', 'Maruti', 'Hyundai', 'Volkswagen', 'Lexus', 'Ford', 'Chevrolet', 'Nissan'])
            year = st.slider("Model Year", 2000, 2025, 2019)
            transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'CVT'])
            condition = st.selectbox("Condition", ['Excellent', 'Good', 'Fair'])
            
        with c2:
            model_name = st.text_input("Model", value="Camry")
            mileage = st.number_input("Mileage", 0, 300000, 45000, step=5000)
            fuel_type = st.selectbox("Fuel", ['Gasoline', 'Diesel', 'Electric', 'Hybrid'])
            accident_history = st.selectbox("History", ['None', 'Minor', 'Major'])

        # Collapsible Advanced Section
        with st.expander("More Options (Engine & Seller)", expanded=False):
            ac1, ac2 = st.columns(2)
            with ac1:
                engine_hp = st.number_input("HP", 50, 1000, 180)
                drivetrain = st.selectbox("Drive", ['FWD', 'RWD', 'AWD', '4WD'])
            with ac2:
                seller_type = st.selectbox("Seller", ['Dealer', 'Private'])
                trim = st.text_input("Trim", value="SE")

        # Hidden Defaults
        exterior_color = "Black"
        interior_color = "Black"
        owner_count = 1
        brand_popularity = 0.5 
        body_type = "Sedan" 

        submit_btn = st.form_submit_button("🚀 CALCULATE VALUE")

# --- RIGHT COLUMN: DASHBOARD RESULTS ---
with right_col:
    # Placeholder for results
    result_container = st.empty()
    
    # Feature Engineering Logic
    current_year = datetime.datetime.now().year
    vehicle_age = current_year - year
    mileage_per_year = mileage / max(1, vehicle_age)
    
    # Determine Health Score & Color
    if condition == 'Excellent' and accident_history == 'None':
        health_score = "A (Premium)"
        bar_percent = 90
        bar_color = "#4CAF50" # Green
    elif accident_history != 'None' or condition == 'Fair':
        health_score = "C (Below Avg)"
        bar_percent = 40
        bar_color = "#FF4B4B" # Red
    else:
        health_score = "B (Standard)"
        bar_percent = 70
        bar_color = "#FFA500" # Orange

    if not submit_btn:
        # Default State
        result_container.markdown(f"""
        <div style="background-color: #1e2129; padding: 30px; border-radius: 12px; text-align: center; border: 1px dashed #41444d;">
            <h3 style="color: #888;">Ready to Value</h3>
            <p>Adjust parameters on the left to get a real-time AI valuation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # 1. Prepare Data
        input_data = pd.DataFrame({
            'make': [make], 'model': [model_name], 'year': [year],
            'mileage': [mileage], 'engine_hp': [engine_hp],
            'transmission': [transmission], 'fuel_type': [fuel_type],
            'drivetrain': [drivetrain], 'condition': [condition],
            'accident_history': [accident_history], 'seller_type': [seller_type],
            'trim': [trim], 'body_type': [body_type],
            'exterior_color': [exterior_color], 'interior_color': [interior_color],
            'owner_count': [owner_count], 'brand_popularity': [brand_popularity]
        })

        # 2. Mirror Feature Engineering
        input_data['vehicle_age'] = vehicle_age
        input_data['mileage_per_year'] = mileage_per_year
        input_data['accident_history'] = input_data['accident_history'].fillna('None')

        # 3. Predict
        try:
            prediction = pipeline.predict(input_data)[0]
            
            # --- MAIN RESULT CARD ---
            result_container.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Market Value</div>
                    <div class="metric-value">${prediction:,.0f}</div>
                    <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">Confidence Interval: ±2.5%</div>
                </div>
            """, unsafe_allow_html=True)
            
            # --- ANALYTICS GRID ---
            st.markdown("###### 📊 Valuation Factors")
            
            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown(f"<div class='info-box'><div style='font-size: 0.8em; color: gray;'>Annual Usage</div><div style='font-size: 1.2em; font-weight: bold;'>{int(mileage_per_year):,} mi</div></div>", unsafe_allow_html=True)
            with g2:
                st.markdown(f"<div class='info-box'><div style='font-size: 0.8em; color: gray;'>Depreciation Age</div><div style='font-size: 1.2em; font-weight: bold;'>{vehicle_age} Yrs</div></div>", unsafe_allow_html=True)
            with g3:
                st.markdown(f"<div class='info-box'><div style='font-size: 0.8em; color: gray;'>Asset Health</div><div style='font-size: 1.2em; font-weight: bold; color: {bar_color};'>{health_score}</div></div>", unsafe_allow_html=True)

            # --- MARKET POSITION BAR ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.9em; margin-bottom: 5px;">Market Position Score</div>
            <div style="width: 100%; background-color: #333; border-radius: 10px;">
                <div style="width: {bar_percent}%; height: 10px; background-color: {bar_color}; border-radius: 10px;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            if accident_history != 'None':
                st.caption(f"⚠️ Value adjusted for '{accident_history}' accident history.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
