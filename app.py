import streamlit as st
import pandas as pd
import joblib
import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Vehicle Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# GLOBAL STYLE (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .main {
        background: radial-gradient(circle at top, #1f2937 0, #020617 45%, #000000 100%) !important;
        color: #e5e7eb;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* App container width */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #f97316, #f97316, #facc15, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        max-width: 600px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.20rem 0.65rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(148,163,184,0.35);
        color: #e5e7eb;
        font-size: 0.75rem;
        margin-bottom: 0.6rem;
    }

    .hero-badge span.dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 4px rgba(34,197,94,0.25);
    }

    /* Glass cards */
    .glass-card {
        background: radial-gradient(circle at top left, rgba(148,163,184,0.18), rgba(15,23,42,0.95));
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 15px 35px rgba(15,23,42,0.7);
    }

    .glass-card-soft {
        background: radial-gradient(circle at top left, rgba(148,163,184,0.16), rgba(15,23,42,0.92));
        border-radius: 18px;
        padding: 1.0rem 1.1rem;
        border: 1px solid rgba(55,65,81,0.7);
        box-shadow: 0 10px 30px rgba(15,23,42,0.65);
    }

    /* Section titles */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.35rem;
    }

    .section-subtitle {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-bottom: 0.8rem;
    }

    /* Metric card */
    .metric-card-main {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 45%, #166534 100%);
        border-radius: 20px;
        padding: 1.2rem 1.1rem;
        color: white;
        text-align: left;
        box-shadow: 0 25px 40px rgba(34,197,94,0.35);
        border: 1px solid rgba(255,255,255,0.18);
    }

    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 3.0rem;
        font-weight: 800;
        margin: 0.1rem 0 0.35rem 0;
        letter-spacing: -0.04em;
    }

    .metric-sub {
        font-size: 0.8rem;
        opacity: 0.95;
    }

    /* Mini chips */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.4rem;
    }

    .chip {
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(15,23,42,0.18);
        background: rgba(255,255,255,0.14);
        font-size: 0.7rem;
        font-weight: 500;
    }

    /* Custom button */
    .stButton>button {
        width: 100%;
        border-radius: 999px;
        height: 3rem;
        border: none;
        background: linear-gradient(90deg, #f97316, #ec4899);
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        box-shadow: 0 14px 30px rgba(249,115,22,0.38);
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.08);
        transform: translateY(-1.5px);
        box-shadow: 0 18px 40px rgba(249,115,22,0.55);
    }

    .stButton>button:focus {
        outline: none;
        box-shadow: 0 0 0 2px #fde68a;
    }

    /* Form labels */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #e5e7eb !important;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stSlider > div > div > div > div {
        border-radius: 10px !important;
    }

    /* Info card before prediction */
    .placeholder-card {
        border-radius: 18px;
        padding: 1.3rem;
        border: 1px dashed rgba(148,163,184,0.5);
        background: radial-gradient(circle at top, rgba(30,64,175,0.2), rgba(15,23,42,0.9));
        color: #e5e7eb;
        font-size: 0.9rem;
    }

    .placeholder-card h3 {
        font-size: 1.0rem;
        margin-bottom: 0.45rem;
    }

    .placeholder-card p {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.6rem;
    }

    .placeholder-steps {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        font-size: 0.8rem;
    }

    .placeholder-step {
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.95);
        border: 1px solid rgba(148,163,184,0.45);
    }

    /* Sub metrics */
    .mini-metric-label {
        font-size: 0.75rem;
        color: #9ca3af;
    }

    .mini-metric-value {
        font-size: 1.0rem;
        font-weight: 600;
        color: #e5e7eb;
    }

    /* Warning styling */
    .impact-alert {
        font-size: 0.83rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# LOAD MODEL ASSETS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load("models/vehicle_price_pipeline.pkl")
        columns = joblib.load("models/input_columns.pkl")
        return pipeline, columns
    except FileNotFoundError:
        try:
            pipeline = joblib.load("vehicle_price_pipeline.pkl")
            columns = joblib.load("input_columns.pkl")
            return pipeline, columns
        except Exception:
            return None, None

pipeline, model_columns = load_assets()

if pipeline is None:
    st.error("⚠️ System Error: Model files missing. Please check repository structure.")
    st.stop()

# -----------------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------------
col_hero_left, col_hero_right = st.columns([2.4, 1.1])

with col_hero_left:
    st.markdown(
        """
        <div class="hero-badge">
            <span class="dot"></span>
            <span>Live Market Estimator · XGBoost-backed</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-title">Used Vehicle Price Studio</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="hero-subtitle">
        Instantly estimate the fair market value of a pre-owned vehicle using a professional
        valuation engine that blends engineering, market behavior and usage patterns.
        Tune configuration on the left and explore insights on the right.
        </p>
        """,
        unsafe_allow_html=True,
    )

with col_hero_right:
    with st.container():
        st.markdown(
            """
            <div class="glass-card-soft">
                <div style="font-size:0.8rem; color:#9ca3af; margin-bottom:0.25rem;">Today’s snapshot</div>
                <div style="display:flex; gap:0.8rem; flex-wrap:wrap;">
                    <div>
                        <div class="mini-metric-label">Model engine</div>
                        <div class="mini-metric-value">XGBoost Pipeline</div>
                    </div>
                    <div>
                        <div class="mini-metric-label">Coverage</div>
                        <div class="mini-metric-value">Multi-brand · Multi-fuel</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.05, 1.4], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN – FORM
# -----------------------------------------------------------------------------
with left_col:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">📋 Vehicle Configuration</div>
            <div class="section-subtitle">
                Start with the key details. Advanced options are available for fine-tuning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")  # small spacer

    with st.form("prediction_form"):
        with st.container():
            st.markdown(
                """
                <div class="section-title" style="margin-top:0.25rem;">Core details</div>
                <div class="section-subtitle">Brand, model, usage & powertrain</div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                make = st.selectbox(
                    "Brand",
                    [
                        "Toyota",
                        "Honda",
                        "BMW",
                        "Maruti",
                        "Hyundai",
                        "Volkswagen",
                        "Lexus",
                        "Ford",
                        "Chevrolet",
                        "Nissan",
                    ],
                )
                year = st.slider("Year of manufacture", 2005, 2025, 2019)
                engine_hp = st.number_input("Engine horsepower", 50, 1000, 180)
                transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
                seller_type = st.selectbox("Seller type", ["Dealer", "Private"])

            with c2:
                model_name = st.text_input("Model name", value="Camry")
                mileage = st.number_input(
                    "Odometer reading (miles)", 0, 300000, 45000, step=1000
                )
                fuel_type = st.selectbox(
                    "Fuel type", ["Gasoline", "Diesel", "Electric", "Hybrid"]
                )
                condition = st.selectbox("Overall condition", ["Excellent", "Good", "Fair"])
                accident_history = st.selectbox(
                    "Accident / damage history", ["None", "Minor", "Major"]
                )

        st.markdown("---")

        with st.expander("⚙️ Advanced options", expanded=False):
            st.markdown(
                """
                <div class="section-subtitle" style="margin-bottom:0.35rem;">
                    These attributes help refine edge cases and high-value vehicles.
                </div>
                """,
                unsafe_allow_html=True,
            )
            d1, d2 = st.columns(2)
            with d1:
                drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD"])
                trim = st.text_input("Trim / variant", value="SE")
            with d2:
                body_type = st.selectbox(
                    "Body style", ["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Van"]
                )
                owner_count = st.number_input("Number of previous owners", 1, 6, 1)

            st.caption(
                "Other cosmetic attributes are assumed but can be wired into the model later."
            )

        # Fixed attributes (can be extended later or exposed in UI)
        exterior_color = "Black"
        interior_color = "Black"
        brand_popularity = 0.5  # placeholder feature for model

        st.markdown("")
        submit_btn = st.form_submit_button("🚀 Run Valuation")

# -----------------------------------------------------------------------------
# RIGHT COLUMN – RESULTS
# -----------------------------------------------------------------------------
with right_col:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">📊 Valuation Analysis</div>
            <div class="section-subtitle">
                See the estimated market value, usage profile and health insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    result_container = st.empty()

    # BEFORE PREDICTION
    if not submit_btn:
        with result_container.container():
            st.markdown(
                """
                <div class="placeholder-card">
                    <h3>Awaiting configuration</h3>
                    <p>
                        Set the vehicle parameters on the left and launch the valuation engine.
                        We’ll compute the fair market value using the trained XGBoost pipeline
                        and highlight the key drivers behind the price.
                    </p>
                    <div class="placeholder-steps">
                        <div class="placeholder-step">① Choose brand & model</div>
                        <div class="placeholder-step">② Set year & mileage</div>
                        <div class="placeholder-step">③ Add condition & damage history</div>
                        <div class="placeholder-step">④ Tap “Run Valuation”</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80",
                caption="AI-powered precision for used vehicle valuation",
                use_column_width=True,
            )

    # AFTER PREDICTION
    if submit_btn:
        # 1. Prepare DataFrame
        input_data = pd.DataFrame(
            {
                "make": [make],
                "model": [model_name],
                "year": [year],
                "mileage": [mileage],
                "engine_hp": [engine_hp],
                "transmission": [transmission],
                "fuel_type": [fuel_type],
                "drivetrain": [drivetrain],
                "condition": [condition],
                "accident_history": [accident_history],
                "seller_type": [seller_type],
                "trim": [trim],
                "body_type": [body_type],
                "exterior_color": [exterior_color],
                "interior_color": [interior_color],
                "owner_count": [owner_count],
                "brand_popularity": [brand_popularity],
            }
        )

        # 2. Feature engineering aligned with training
        current_year = datetime.datetime.now().year
        input_data["vehicle_age"] = current_year - input_data["year"]
        # protect against div by zero
        input_data["mileage_per_year"] = input_data["mileage"] / input_data[
            "vehicle_age"
        ].replace(0, 1)
        input_data["accident_history"] = input_data["accident_history"].fillna("None")

        # Align columns with training if available
        if model_columns is not None:
            # Reindex to expected training columns; missing cols filled with 0
            input_data = input_data.reindex(columns=model_columns, fill_value=0)

        try:
            # 3. Predict
            prediction = float(pipeline.predict(input_data)[0])

            # basic qualitative banding
            age = int(input_data.get("vehicle_age", pd.Series([0])).iloc[0])
            mp_year = float(input_data.get("mileage_per_year", pd.Series([0])).iloc[0])

            if age <= 3 and mp_year < 14000 and accident_history == "None":
                pricing_band = "Premium resale segment"
                band_note = "Strong residual value driven by low age & clean history."
            elif age <= 7 and accident_history in ["None", "Minor"]:
                pricing_band = "Mainstream fair value"
                band_note = "Healthy balance between depreciation and usability."
            else:
                pricing_band = "Value / budget tier"
                band_note = "Price is shaped more by age, mileage and risk factors."

            # MAIN CARD
            with result_container.container():
                st.markdown(
                    f"""
                    <div class="metric-card-main">
                        <div class="metric-label">ESTIMATED FAIR MARKET VALUE</div>
                        <div class="metric-value">${prediction:,.0f}</div>
                        <div class="metric-sub">
                            {make} {model_name} · {year} · {int(mileage):,} mi · {fuel_type}
                        </div>
                        <div class="chip-row">
                            <div class="chip">{pricing_band}</div>
                            <div class="chip">Condition: {condition}</div>
                            <div class="chip">Accident: {accident_history}</div>
                            <div class="chip">{transmission} · {drivetrain}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.success("Valuation complete • Model successfully evaluated the configuration.")

                # SECONDARY METRICS
                st.markdown("#### 💡 Valuation Drivers")

                m1, m2, m3 = st.columns(3)
                m1.metric("Depreciation age", f"{age} years")
                m2.metric("Usage intensity", f"{int(mp_year):,} mi / year")

                if condition == "Excellent" and accident_history == "None":
                    health_score = "A+"
                elif accident_history != "None":
                    health_score = "C"
                else:
                    health_score = "B"

                m3.metric("Vehicle health grade", health_score)

                # Narrative explanation
                st.markdown(
                    f"""
                    **Why this price?**

                    - Age and usage pattern suggest a **{pricing_band.lower()}** profile.  
                    - Condition is reported as **{condition}**, with accident history marked as **{accident_history}**.  
                    - Powertrain: **{engine_hp} HP {fuel_type}**, **{transmission}** transmission and **{drivetrain}** drivetrain.  
                    - Ownership trail shows **{owner_count}** recorded owner(s), which also affects buyer confidence.
                    """
                )

                if accident_history != "None":
                    st.warning(
                        f"📉 **Impact alert:** Valuation includes a penalty for `{accident_history}` accident history. "
                        "Severe structural damage or poor repair quality can push real-world prices further down.",
                        icon="⚠️",
                    )

                # Micro insight row
                st.markdown("---")
                col_ins1, col_ins2 = st.columns([1.2, 1.0])
                with col_ins1:
                    st.markdown(
                        f"""
                        **Market tip:**  
                        For a {year} {make} {model_name}, keeping mileage below
                        roughly **{int((age + 1) * 12000):,} miles** and maintaining service
                        records can support a higher closing price.
                        """
                    )
                with col_ins2:
                    st.info(
                        "This valuation is an AI-assisted estimate. Local market conditions, city, dealer margins "
                        "and seasonal demand can shift actual selling price.",
                        icon="💡",
                    )

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
