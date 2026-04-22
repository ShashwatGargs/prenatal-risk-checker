import streamlit as st
import joblib
import pandas as pd
import numpy as np


# PAGE CONFIG

st.set_page_config(
    page_title="Pregnancy Risk Checker",
    page_icon="🤰",
    layout="centered"
)


# LOAD MODEL

model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("label_encoder.pkl")


# HEADER

st.title("🤰 Pregnancy Risk Checker")
st.caption("Simple tool to estimate pregnancy risk level (no medical knowledge needed)")

st.markdown("---")


# USER INPUT


st.markdown("## 📝 Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Your age", 18, 45, 25)
    preg_count = st.slider("Times pregnant", 1, 10, 1)
    weeks = st.slider("Weeks pregnant", 20, 42, 30)

with col2:
    weight = st.slider("Weight (kg)", 40, 120, 60)
    height = st.slider("Height (feet, e.g. 5.3)", 4.0, 7.0, 5.3)

st.markdown("## 🏥 Health Info")

col3, col4 = st.columns(2)

with col3:
    bp = st.selectbox("High blood pressure?", ["No", "Yes"])
    sugar = st.selectbox("High blood sugar (diabetes)?", ["No", "Yes"])

with col4:
    weakness = st.selectbox("Feeling weak / low iron?", ["No", "Yes"])
    heart = st.slider("Baby heart rate (if known)", 100, 180, 130)


# CONVERT INPUT

bp_val = 1 if bp == "Yes" else 0
sugar_val = 1 if sugar == "Yes" else 0
anemia_val = 1 if weakness == "Yes" else 0

data = pd.DataFrame([{
    "Age": age,
    "Gravida": preg_count,
    "TT": 2,
    "Gestation": weeks,
    "Weight": weight,
    "Height": height,
    "Anemia": anemia_val,
    "Jaundice": 0,
    "FetalPosition": 0,
    "FetalMovement": 0,
    "FetalHeartRate": heart,
    "Albumin": 0,
    "Sugar": sugar_val,
    "VDRL": 0,
    "HBsAg": 0,
    "SystolicBP": 120 if bp_val == 0 else 140,
    "DiastolicBP": 80 if bp_val == 0 else 90
}])

# PREDICTION

if st.button("🔍 Check Risk"):
    pred = model.predict(data)
    result = encoder.inverse_transform(pred)[0]

    proba = model.predict_proba(data)[0]
    confidence = max(proba) * 100

    st.markdown("---")
    st.subheader("📊 Result")

    # Risk display
    if result.lower() == "yes":
        st.error("🔴 Higher Risk Detected")
        st.progress(90)
    else:
        st.success("🟢 Lower Risk")
        st.progress(30)

    st.write(f"**Confidence:** {confidence:.1f}%")

    # EXPLANATION

    st.markdown("### 🧠 What this means")

    if result.lower() == "yes":
        st.write("This does NOT mean something is wrong, but extra care is recommended.")
        st.write("Possible influencing factors:")
        st.write("- Blood pressure")
        st.write("- Blood sugar")
        st.write("- Pregnancy history")
    else:
        st.write("Things look normal based on your inputs.")
        st.write("Continue regular care and checkups.")

    # FEATURE IMPORTANCE

    st.markdown("### 📈 Key Factors")

    importance = model.feature_importances_
    features = data.columns

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))

    # ADVICE

    st.markdown("### 💡 What you should do")

    if result.lower() == "yes":
        st.write("👉 Visit a doctor soon")
        st.write("👉 Monitor blood pressure regularly")
        st.write("👉 Maintain a healthy diet")
    else:
        st.write("👉 Continue regular checkups")
        st.write("👉 Eat a balanced diet")
        st.write("👉 Stay active")

# FOOTER

st.markdown("---")

st.warning("""
⚠️ This tool is for awareness only and is NOT a medical diagnosis.
Always consult a qualified doctor for proper medical advice.
""")

st.markdown("🔗 Share this tool with others:")
st.code("https://prenatal-risk-checker-fv6e9rsnishvzu682aqrjt.streamlit.app/")