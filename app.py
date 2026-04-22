import streamlit as st
import joblib
import pandas as pd

# Configure the app page (title, icon, layout)
st.set_page_config(
    page_title="Pregnancy Risk Checker",
    page_icon="🤰",
    layout="centered"
)

# Load trained ML model and label encoder
model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# App title and description
st.title("🤰 Pregnancy Risk Checker")
st.caption("Simple tool to estimate pregnancy risk level (no medical knowledge needed)")

st.markdown("---")

# Section: User basic details
st.markdown("## 📝 Your Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Your age", 18, 45, 25)  # Age restricted to 18+
    preg_count = st.slider("Times pregnant", 1, 10, 1)
    weeks = st.slider("Weeks pregnant", 20, 42, 30)

with col2:
    weight = st.slider("Weight (kg)", 40, 120, 60)
    height = st.slider("Height (feet, e.g. 5.3)", 4.0, 7.0, 5.3)

# Section: Health-related inputs
st.markdown("## 🏥 Health Info")

col3, col4 = st.columns(2)

with col3:
    bp = st.selectbox("High blood pressure?", ["No", "Yes"])
    sugar = st.selectbox("High blood sugar (diabetes)?", ["No", "Yes"])

with col4:
    weakness = st.selectbox("Feeling weak / low iron?", ["No", "Yes"])
    heart = st.slider("Baby heart rate (if known)", 100, 180, 130)

# Convert user-friendly inputs into numerical values for the model
bp_val = 1 if bp == "Yes" else 0
sugar_val = 1 if sugar == "Yes" else 0
anemia_val = 1 if weakness == "Yes" else 0

# Create a dataframe matching the model's expected input format
data = pd.DataFrame([{
    "Age": age,
    "Gravida": preg_count,
    "TT": 2,  # Default assumed value
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

# Run prediction when button is clicked
if st.button("🔍 Check Risk"):
    pred = model.predict(data)
    result = encoder.inverse_transform(pred)[0]

    # Get prediction confidence
    proba = model.predict_proba(data)[0]
    confidence = max(proba) * 100

    st.markdown("---")
    st.subheader("📊 Result")

    # Display result visually
    if result.lower() == "yes":
        st.error("🔴 Higher Risk Detected")
        st.progress(90)
    else:
        st.success("🟢 Lower Risk")
        st.progress(30)

    # Show confidence percentage
    st.write(f"**Confidence:** {confidence:.1f}%")

    # Explain result in simple language
    st.markdown("### 🧠 What this means")

    if result.lower() == "yes":
        st.write("This does NOT mean something is wrong, but extra care is recommended.")
        st.write("- Blood pressure")
        st.write("- Blood sugar")
        st.write("- Pregnancy history")
    else:
        st.write("Things look normal based on your inputs.")

    # Provide simple actionable advice
    st.markdown("### 💡 What you should do")

    if result.lower() == "yes":
        st.write("👉 Visit a doctor soon")
        st.write("👉 Monitor health regularly")
    else:
        st.write("👉 Continue regular checkups")

# Footer section
st.markdown("---")

# Disclaimer for safety
st.warning("⚠️ This tool is for awareness only and is NOT a medical diagnosis.")

# Shareable app link
st.code("https://prenatal-risk-checker-fv6e9rsnishvzu682aqrjt.streamlit.app/")