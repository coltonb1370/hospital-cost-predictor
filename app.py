
import streamlit as st
import pandas as pd
import joblib

# Load preprocessor and model
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("preprocessor_pipeline_v2.pkl")
    model = joblib.load("xgboost_model.pkl")
    return preprocessor, model

preprocessor, model = load_assets()

# Streamlit app layout
st.title("🏥 Hospital Billing Cost Predictor")
st.markdown("Enter patient and treatment details to estimate the expected hospital billing cost.")

# Sidebar inputs
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Patient Age", 0, 100, 40)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
blood_type = st.sidebar.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
condition = st.sidebar.selectbox("Medical Condition", ["Asthma", "Flu", "Infections", "Cancer", "Heart Disease"])
hospital = st.sidebar.selectbox("Hospital", [
    "Northwestern Memorial Hospital", 
    "UI Health (University of Illinois Hospital)"
])
insurance = st.sidebar.selectbox("Insurance Provider", ["Blue Cross", "Aetna", "Cigna", "UnitedHealthcare"])
admission = st.sidebar.selectbox("Admission Type", ["Emergency", "Routine", "Elective"])
medication = st.sidebar.selectbox("Medication", ["Azithromycin", "Tamiflu", "Cisplatin", "Prednisone", "Beta-blockers"])
test_result = st.sidebar.selectbox("Test Results", ["Normal", "Abnormal", "Inconclusive"])
stay = st.sidebar.slider("Length of Stay (days)", 1, 60, 3)

# Create input DataFrame
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

# Predict cost
if st.button("Predict Billing Amount"):
    try:
        processed_input = preprocessor.transform(input_data)
        prediction = model.predict(processed_input)
        st.success(f"💰 Estimated Billing Amount: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Help section
with st.expander("ℹ️ Help"):
    st.markdown("""
    This tool estimates hospital billing cost based on patient demographics, insurance type, and clinical details.
    
    - **Step 1**: Fill in patient and visit details from the sidebar.
    - **Step 2**: Click **Predict Billing Amount**.
    - **Step 3**: View the estimated hospital charge.

    **Note**: This tool uses historical data and machine learning to provide an estimate. It is not a guarantee of actual costs.
    """)
