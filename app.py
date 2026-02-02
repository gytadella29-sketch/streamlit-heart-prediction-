import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide"
)

# ==================================================
# DARK MODE TOGGLE
# ==================================================
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top, #0b1d33, #050b14);
        color: #ffffff;
    }
    h1,h2,h3,h4,h5,h6,p,label,span {
        color: white !important;
    }
    .glass {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(18px);
        border-radius: 24px;
        padding: 35px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }
    .accent {
        color: #4da3ff;
        font-weight: 700;
    }
    .stButton button {
        background: linear-gradient(135deg, #1e90ff, #4da3ff);
        color: white;
        border-radius: 16px;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
    }
    .risk-high { color: #ff4b4b; font-weight: 800; }
    .risk-mid { color: #ffb347; font-weight: 800; }
    .risk-low { color: #4cd964; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ==================================================
# HERO / LANDING
# ==================================================
st.markdown("""
<div class="glass">
    <h1 style="text-align:center;">❤️ HeartCare AI</h1>
    <h3 style="text-align:center;">Smart Heart Disease Risk Prediction System</h3>
    <p style="text-align:center; font-size:18px;">
        Sistem pendukung keputusan medis berbasis<br>
        <span class="accent">Stacking Ensemble Machine Learning</span>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================================================
# FEATURES
# ==================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="glass">
    <h4>🧠 AI Intelligence</h4>
    <p>Menggunakan ensemble learning untuk prediksi risiko yang lebih stabil dan akurat.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass">
    <h4>📊 Risk Analytics</h4>
    <p>Menampilkan probabilitas, confidence level, dan interpretasi medis secara visual.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="glass">
    <h4>🩺 Clinical Support</h4>
    <p>Membantu tenaga medis dan pasien dalam pengambilan keputusan awal.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================================================
# INPUT SECTION
# ==================================================
st.markdown("## 📝 Patient Clinical Data")

left, right = st.columns(2)

with left:
    age = st.number_input("Age (years)", min_value=0, step=1, value=None)
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0, value=None)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, value=None)
    max_hr = st.number_input("Maximum Heart Rate", min_value=0, value=None)

with right:
    sex = st.selectbox("Sex", ["Select", "M", "F"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Select", 0, 1])
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Select", "Y", "N"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", value=None)

st.markdown("---")

# ==================================================
# PREDICTION
# ==================================================
if st.button("🔍 Analyze Heart Risk", use_container_width=True):

    if (
        None in [age, resting_bp, cholesterol, max_hr, oldpeak] or
        "Select" in [sex, fasting_bs, exercise_angina]
    ):
        st.warning("⚠️ Please complete all patient data.")
        st.stop()

    data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "FastingBS": int(fasting_bs),
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0
    }

    df = pd.DataFrame([data]).reindex(columns=columns)
    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0][1]

    # ==================================================
    # RISK LEVEL
    # ==================================================
    if prob >= 0.7:
        risk = "HIGH RISK"
        risk_class = "risk-high"
    elif prob >= 0.4:
        risk = "MODERATE RISK"
        risk_class = "risk-mid"
    else:
        risk = "LOW RISK"
        risk_class = "risk-low"

    confidence = abs(prob - 0.5) * 2

    # ==================================================
    # RESULT DASHBOARD
    # ==================================================
    st.markdown("""
    <div class="glass">
        <h2>📊 Prediction Result</h2>
        <h1 class="{0}">{1}</h1>
    </div>
    """.format(risk_class, risk), unsafe_allow_html=True)

    st.metric("Risk Probability", f"{prob:.2%}")
    st.progress(int(prob * 100))

    st.markdown("### 🧠 Model Confidence")
    st.write(f"Confidence Score: **{confidence:.2%}**")

    st.markdown("### 📝 Medical Interpretation")
    if risk == "HIGH RISK":
        st.write(
            "The model indicates a **high risk of heart disease**. "
            "Immediate medical evaluation is strongly recommended."
        )
    elif risk == "MODERATE RISK":
        st.write(
            "The model indicates a **moderate risk of heart disease**. "
            "Lifestyle improvements and further clinical checks are advised."
        )
    else:
        st.write(
            "The model indicates a **low risk of heart disease**. "
            "Maintain a healthy lifestyle and regular medical check-ups."
        )

    st.info(
        "⚠️ This system is a **clinical decision support tool** "
        "and does not replace professional medical diagnosis."
    )

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("© HeartCare AI | Advanced Medical Decision Support System")