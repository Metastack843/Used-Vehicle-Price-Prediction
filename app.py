import streamlit as st
import pandas as pd
import joblib
import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Vehicle Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# GLOBAL STYLE (COMPACT)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top, #020617 0, #020617 40%, #000000 100%) !important;
        color: #e5e7eb;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        max-width: 1250px;
    }

    /* HERO */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #f97316, #f97316, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 0.82rem;
        color: #9ca3af;
        max-width: 520px;
        line-height: 1.3;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.35);
        color: #e5e7eb;
        font-size: 0.72rem;
        margin-bottom: 0.35rem;
    }
    .hero-badge span.dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 3px rgba(34,197,94,0.3);
    }

    /* CARDS */
    .glass-card {
        background: radial-gradient(circle at top left, rgba(148,163,184,0.18), rgba(15,23,42,0.96));
        border-radius: 16px;
        padding: 0.9rem 1.0rem 0.9rem 1.0rem;
        border: 1px solid rgba(55,65,81,0.7);
        box-shadow: 0 12px 28px rgba(15,23,42,0.75);
    }

    .section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.15rem;
    }
    .section-subtitle {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.55rem;
    }

    /* MAIN METRIC CARD */
    .metric-card-main {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 45%, #166534 100%);
        border-radius: 16px;
        padding: 0.9rem 1.0rem;
        color: white;
        text-align: left;
        box-shadow: 0 18px 32px rgba(34,197,94,0.4);
        border: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 0.7rem;
    }
    .metric-label {
        font-size: 0.8rem;
        opacity: 0.9;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.05rem 0 0.25rem 0;
        letter-spacing: -0.04em;
    }
    .metric-sub {
        font-size: 0.78rem;
        opacity: 0.95;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-top: 0.4rem;
    }
    .chip {
        padding: 0.18rem 0.5rem;
        border-radius: 999px;
        border: 1px solid rgba(15,23,42,0.2);
        background: rgba(255,255,255,0.15);
        font-size: 0.7rem;
        font-weight: 500;
    }

    /* PLACEHOLDER */
    .placeholder-card {
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        border: 1px dashed rgba(148,163,184,0.5);
        background: radial-gradient(circle at top, rgba(30,64,175,0.22), rgba(15,23,42,0.96));
        font-size: 0.78rem;
        margin-bottom: 0.5rem;
    }
    .placeholder-steps {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.3rem;
    }
    .placeholder-step {
        padding: 0.15rem 0.45rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(148,163,184,0.5);
        font-size: 0.7rem;
    }

    /* BUTTON */
    .stButton>button {
        width: 100%;
        border-radius: 999px;
        height: 2.6rem;
        border: none;
        background: linear-gradient(90deg, #f97316, #ec4899);
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        box-shadow: 0 12px 26px rgba(249,115,22,0.4);
        transition: all 0.22s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(249,115,22,0.55);
    }

    /* LABELS & INPUTS */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #e5e7eb !important;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    .mini-metric-label {
        font-size: 0.72rem;
        color: #9ca3af;
    }
    .mini-metric-value {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LOAD MODEL ASSETS (same behavior as your original code, but path-safe)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    base_dir = Path(__file__).resolve().parent

    model_candidates = [
        base_dir / "models" / "vehicle_price_pipeline.pkl",
        base_dir / "vehicle_price_pipeline.pkl",
    ]
    cols_candidates = [
        base_dir / "models" / "input_columns.pkl",
        base_dir / "input_columns.pkl",
    ]

    model_file = next((p for p in model_candidates if p.exists()), None)
    cols_file = next((p for p in cols_candidates if p.exists()), None)

    if model_file is None or cols_file is None:
        return None, None

    pipeline = joblib.load(model_file)
    columns = joblib.load(cols_file)
    return pipeline, columns


pipeline, model_columns = load_assets()
if pipeline is None:
    st.error("⚠️ System Error: Model files missing. Please check repository structure.")
    st.stop()

# -----------------------------------------------------------------------------
# HERO (VERY COMPACT)
# -----------------------------------------------------------------------------
col_hero_left, col_hero_right = st.columns([2.2, 1.0])

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
        Configure the vehicle on the left and instantly see its fair market value with
        AI-assisted insights on the right — all in a single screen.
        </p>
        """,
        unsafe_allow_html=True,
    )

with col_hero_right:
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
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
# MAIN TWO-COLUMN LAYOUT (no extra headers, just one card per side)
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.1, 1.3], gap="large")

# LEFT: FORM
with left_col:
    with st.container():
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">📋 Vehicle Configuration</div>
                <div class="section-subtitle">
                    Keep it simple: set brand, usage and condition. Advanced options are tucked below.
                </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
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

            with st.expander("⚙️ Advanced options", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD"])
                    trim = st.text_input("Trim / variant", value="SE")
                with d2:
                    body_type = st.selectbox(
                        "Body style",
                        ["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Van"],
                    )
                    owner_count = st.number_input("Number of previous owners", 1, 6, 1)

            exterior_color = "Black"
            interior_color = "Black"
            brand_popularity = 0.5

            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🚀 Run Valuation")

        st.markdown("</div>", unsafe_allow_html=True)  # close glass-card

# RIGHT: RESULT PANEL
with right_col:
    with st.container():
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">📊 Valuation Analysis</div>
                <div class="section-subtitle">
                    Estimated market value, usage profile and health in one glance.
                </div>
            """,
            unsafe_allow_html=True,
        )

        result_container = st.empty()

        # BEFORE PREDICTION: very small placeholder, no big image
        if not submit_btn:
            with result_container.container():
                st.markdown(
                    """
                    <div class="placeholder-card">
                        Configure the vehicle and run the valuation. The price card and insights will appear here.
                        <div class="placeholder-steps">
                            <div class="placeholder-step">① Brand & model</div>
                            <div class="placeholder-step">② Year & mileage</div>
                            <div class="placeholder-step">③ Condition & accidents</div>
                            <div class="placeholder-step">④ Tap “Run Valuation”</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # AFTER PREDICTION
        if submit_btn:
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

            current_year = datetime.datetime.now().year
            input_data["vehicle_age"] = current_year - input_data["year"]
            input_data["mileage_per_year"] = input_data["mileage"] / input_data[
                "vehicle_age"
            ].replace(0, 1)
            input_data["accident_history"] = input_data["accident_history"].fillna("None")

            if model_columns is not None:
                input_data = input_data.reindex(columns=model_columns, fill_value=0)

            try:
                prediction = float(pipeline.predict(input_data)[0])
                age = int(input_data.get("vehicle_age", pd.Series([0])).iloc[0])
                mp_year = float(input_data.get("mileage_per_year", pd.Series([0])).iloc[0])

                if age <= 3 and mp_year < 14000 and accident_history == "None":
                    pricing_band = "Premium resale"
                elif age <= 7 and accident_history in ["None", "Minor"]:
                    pricing_band = "Fair value"
                else:
                    pricing_band = "Budget / value"

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

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Depreciation age", f"{age} yrs")
                    m2.metric("Usage intensity", f"{int(mp_year):,} mi/yr")

                    if condition == "Excellent" and accident_history == "None":
                        health_score = "A+"
                    elif accident_history != "None":
                        health_score = "C"
                    else:
                        health_score = "B"

                    m3.metric("Health grade", health_score)

                    st.caption(
                        "AI-assisted estimate. Local market, city, dealer margin and season can shift the final selling price."
                    )

                    if accident_history != "None":
                        st.warning(
                            f"📉 Accident history marked as `{accident_history}` may push real-world offers below this estimate.",
                            icon="⚠️",
                        )

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)  # close glass-card
