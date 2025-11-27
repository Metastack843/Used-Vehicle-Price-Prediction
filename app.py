import streamlit as st
import pandas as pd
import joblib
import datetime

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION (Wide Layout for "One Page" Feel)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Vehicle Price Prediction",
    page_icon="🚘",
    layout="wide",  # Uses full screen width
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS (The "Modern Look" Engine)
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Card Container Styling */
    .stForm {
        background-color: #262730;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #41444d;
    }
    
    /* Result Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #FF4B4B 0%, #bd3030 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(255, 75, 75, 0.2);
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
    }
    .metric-label {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 5px;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
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
            # Fallback for local testing
            pipeline = joblib.load('vehicle_price_pipeline.pkl')
            columns = joblib.load('input_columns.pkl')
            return pipeline, columns
        except:
            return None, None

pipeline, model_columns = load_assets()

if pipeline is None:
    st.error("⚠️ System Error: Model files missing. Please check repository structure.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. APP LAYOUT
# ------------------------------------------------------------------------------

# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🚘 Used Vehicle Price Prediction")
    st.caption("Professional Market Valuation Engine powered by XGBoost")

st.markdown("---")

# Main Split Layout
left_col, right_col = st.columns([1, 1.5], gap="large")

# --- LEFT COLUMN: CONTROLS (The "Card") ---
with left_col:
    st.markdown("### 📋 Vehicle Configuration")
    
    with st.form("prediction_form"):
        # Row 1
        c1, c2 = st.columns(2)
        with c1:
            make = st.selectbox("Brand", ['Toyota', 'Honda', 'BMW', 'Maruti', 'Hyundai', 'Volkswagen', 'Lexus', 'Ford', 'Chevrolet', 'Nissan'])
            year = st.slider("Year", 2005, 2025, 2019)
            engine_hp = st.number_input("Horsepower", 50, 1000, 180)
            transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'CVT'])
            seller_type = st.selectbox("Seller", ['Dealer', 'Private'])
            
        with c2:
            model_name = st.text_input("Model", value="Camry")
            mileage = st.number_input("Mileage", 0, 300000, 45000, step=1000)
            fuel_type = st.selectbox("Fuel", ['Gasoline', 'Diesel', 'Electric', 'Hybrid'])
            condition = st.selectbox("Condition", ['Excellent', 'Good', 'Fair'])
            accident_history = st.selectbox("Damage", ['None', 'Minor', 'Major'])

        # Advanced/Hidden inputs (Accordion to save space)
        with st.expander("Advanced Options"):
            drivetrain = st.selectbox("Drivetrain", ['FWD', 'RWD', 'AWD', '4WD'])
            trim = st.text_input("Trim", value="SE")
            
        # Defaults
        exterior_color = "Black"
        interior_color = "Black"
        owner_count = 1
        brand_popularity = 0.5 
        body_type = "Sedan" 

        submit_btn = st.form_submit_button("🚀 Run Valuation")

# --- RIGHT COLUMN: RESULTS (The "Dashboard") ---
with right_col:
    st.markdown("### 📊 Valuation Analysis")
    
    # Placeholder for results
    result_container = st.empty()
    
    # Default State (Before Calculation)
    if not submit_btn:
        result_container.info("👈 Configure vehicle details on the left to generate a valuation.")
        st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", caption="AI-Powered Precision", use_column_width=True)

    # Calculation State
    if submit_btn:
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

        # 2. Feature Engineering
        current_year = datetime.datetime.now().year
        input_data['vehicle_age'] = current_year - input_data['year']
        input_data['mileage_per_year'] = input_data['mileage'] / input_data['vehicle_age'].replace(0, 1)
        input_data['accident_history'] = input_data['accident_history'].fillna('None')

        # 3. Predict
        try:
            prediction = pipeline.predict(input_data)[0]
            
            # --- THE BEAUTIFUL RESULT CARD ---
            result_container.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ESTIMATED MARKET VALUE</div>
                    <div class="metric-value">${prediction:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("Valuation Complete!")
            
            # 4. Contextual Insights (The "Why")
            st.markdown("#### 💡 Valuation Drivers")
            
            # 3-Column Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Depreciation Age", f"{input_data['vehicle_age'][0]} Years")
            m2.metric("Usage Intensity", f"{int(input_data['mileage_per_year'][0]):,} mi/yr")
            
            # Dynamic Health Score logic
            if condition == 'Excellent' and accident_history == 'None':
                health_score = "A+"
                color = "green"
            elif accident_history != 'None':
                health_score = "C"
                color = "red"
            else:
                health_score = "B"
                color = "orange"
                
            m3.metric("Vehicle Health Grade", health_score)
            
            if accident_history != 'None':
                st.warning(f"📉 **Impact Alert:** Valuation includes a penalty for '{accident_history}' accident history.")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
