import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import xgboost as xgb

# Load model and preprocessor
def load_model(path):
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model

preprocessor = joblib.load("preprocessor_pipeline_v2.pkl")
model = load_model("xgboost_model_v2.json")

# Streamlit UI
st.title("🏥 Hospital Billing Cost Predictor")
st.markdown("Enter patient and treatment details to estimate hospital billing cost.")

# Sidebar input
st.sidebar.header("Patient Info")
age = st.sidebar.slider("Age", 0, 100, 40)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
blood_type = st.sidebar.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
condition = st.sidebar.selectbox("Condition", ["Asthma", "Flu", "Infections", "Cancer", "Heart Disease"])
hospital = st.sidebar.selectbox("Hospital", [
    "Northwestern Memorial Hospital", 
    "UI Health (University of Illinois Hospital)"
])
insurance = st.sidebar.selectbox("Insurance", ["Blue Cross", "Aetna", "Cigna", "UnitedHealthcare"])
admission = st.sidebar.selectbox("Admission Type", ["Emergency", "Routine", "Elective"])
medication = st.sidebar.selectbox("Medication", ["Azithromycin", "Tamiflu", "Cisplatin", "Prednisone", "Beta-blockers"])
test_result = st.sidebar.selectbox("Test Result", ["Normal", "Abnormal", "Inconclusive"])
stay = st.sidebar.slider("Length of Stay", 1, 60, 3)

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Blood Type": [blood_type],
    "Medical Condition": [condition],
    "Hospital": [hospital],
    "Insurance Provider": [insurance],
    "Admission Type": [admission],
    "Medication": [medication],
    "Test Results": [test_result],
    "Length of Stay": [stay]
})

# Prediction
if st.button("Predict Billing Amount"):
    try:
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)
        st.success(f"💰 Estimated Billing Amount: ${prediction[0]:,.2f}")

        # Plot
        st.subheader("📈 Evaluation Plot")
        image = Image.open("model_evaluation_plot.png")
        st.image(image, caption="Predicted vs. Actual Billing Amount")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Help section
with st.expander("ℹ️ Help"):
    st.markdown(\"""
    Use the sidebar to enter patient data, then click **Predict Billing Amount**.
    You will receive a predicted treatment cost and a visualization of model accuracy.
    \""")