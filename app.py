import streamlit as st
import joblib
import numpy as np
import math

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💼",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #7a7a9a;
    font-size: 1rem;
    font-weight: 300;
}
.accent { color: #6c63ff; }

/* Card */
.card {
    background: #13131f;
    border: 1px solid #1f1f35;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6c63ff;
    margin-bottom: 1.2rem;
}

/* Inputs */
.stNumberInput label, .stSelectbox label, .stSlider label {
    color: #b0b0c8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
.stNumberInput input {
    background: #1a1a2e !important;
    border: 1px solid #2a2a45 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
.stNumberInput input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.15) !important;
}

/* Toggle / checkbox */
.stCheckbox label {
    color: #b0b0c8 !important;
    font-size: 0.88rem !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: opacity 0.2s;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    opacity: 0.88;
}

/* Result boxes */
.result-high {
    background: linear-gradient(135deg, #0f2a1a, #0a1f12);
    border: 1px solid #1a5c30;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #1f0f0f, #1a0a0a);
    border: 1px solid #5c1a1a;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-sub { color: #7a7a9a; font-size: 0.88rem; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1f1f35;
    margin: 1.5rem 0;
}

/* Metric row */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #6c63ff;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #5a5a7a;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Income <span class="accent">Predictor</span></h1>
    <p>ML-powered prediction · Census Income Dataset · XGBoost Top-5 Features</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📊 Input Features</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=17, max_value=90, value=35,
        help="Age of the individual"
    )
    education_num = st.number_input(
        "Education Level (1–16)",
        min_value=1, max_value=16, value=10,
        help="1=Preschool → 16=Doctorate"
    )

with col2:
    capital_gain = st.number_input(
        "Capital Gain (USD)",
        min_value=0, max_value=100000, value=0, step=500,
        help="Annual capital gain income"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    married = st.checkbox("Married (Civilian Spouse)", value=False)
    is_husband = st.checkbox("Relationship: Husband", value=False)

st.markdown('</div>', unsafe_allow_html=True)

# Education level reference
edu_map = {
    1:"Preschool", 2:"1st-4th", 3:"5th-6th", 4:"7th-8th", 5:"9th",
    6:"10th", 7:"11th", 8:"12th", 9:"HS-grad", 10:"Some-college",
    11:"Assoc-voc", 12:"Assoc-acdm", 13:"Bachelors", 14:"Masters",
    15:"Prof-school", 16:"Doctorate"
}
st.caption(f"🎓 Education level {education_num} = **{edu_map[education_num]}**")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("⚡ PREDICT INCOME"):

    # Apply log transform to capital-gain (same as training)
    capital_gain_log = math.log(capital_gain + 1)

    features = np.array([[
        capital_gain_log,
        int(married),
        age,
        education_num,
        int(is_husband)
    ]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = max(proba) * 100

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
            <div class="result-icon">💰</div>
            <div class="result-label" style="color:#4ade80;">Income &gt; $50K</div>
            <div class="result-sub">High income bracket predicted</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <div class="result-icon">📉</div>
            <div class="result-label" style="color:#f87171;">Income ≤ $50K</div>
            <div class="result-sub">Standard income bracket predicted</div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics row
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-val">{confidence:.1f}%</div>
            <div class="metric-lbl">Confidence</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{proba[1]*100:.1f}%</div>
            <div class="metric-lbl">P(&gt;50K)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{proba[0]*100:.1f}%</div>
            <div class="metric-lbl">P(≤50K)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class='divider'>
<div style="text-align:center; color:#3a3a5a; font-size:0.78rem; font-family:'Space Mono',monospace;">
    XGBoost · Top-5 Features · Census Income Dataset
</div>
""", unsafe_allow_html=True)
