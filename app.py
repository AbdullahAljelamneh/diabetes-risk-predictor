import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter the patient's clinical values below to assess diabetes risk.")

# ── Input form ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=20, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=5, max_value=100, value=29)
    insulin = st.number_input("Insulin (μU/mL)", min_value=10, max_value=900, value=130)

with col2:
    bmi = st.number_input("BMI", min_value=15.0, max_value=70.0, value=32.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=0.47)
    age = st.number_input("Age", min_value=21, max_value=90, value=33)

# ── Feature engineering (must match training) ────────────
bmi_age = bmi * age
glucose_insulin_ratio = glucose / insulin
glucose_bmi = glucose * bmi

features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                       insulin, bmi, dpf, age,
                       bmi_age, glucose_insulin_ratio, glucose_bmi]])

# ── Load model and scaler ────────────────────────────────
import joblib
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

features_scaled = scaler.transform(features)

# ── Predict ──────────────────────────────────────────────
if st.button("Predict Risk", type="primary"):
    prob = model.predict_proba(features_scaled)[0][1]
    pred = model.predict(features_scaled)[0]

    st.markdown("---")

    if pred == 1:
        st.error(f"⚠️ High Diabetes Risk Detected")
    else:
        st.success(f"✅ Low Diabetes Risk")

    st.metric(label="Diabetes Probability", value=f"{prob*100:.1f}%")

    # Risk bar
    st.progress(float(prob))

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and is not a medical diagnosis.")