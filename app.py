
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.markdown("""
    Use the sidebar to enter patient data, then click **Predict Billing Amount**.
    You will receive a predicted treatment cost and a visualization of model accuracy.
    """)

    # Load your dataset
df = pd.read_csv("modified_healthcare_dataset.csv")  # replace with your actual path or source

st.subheader("📊 Summary Statistics")

# Numerical summary for selected columns
st.markdown("### Numerical Features Summary")
numeric_cols = ["Age", "Billing Amount", "Length of Stay"]
st.dataframe(df[numeric_cols].describe().transpose())

# Categorical summary: Hospital
st.markdown("### Categorical Feature Summary: Hospital")
hospital_counts = df["Hospital"].value_counts().reset_index()
hospital_counts.columns = ["Hospital", "Count"]
st.dataframe(hospital_counts)

st.subheader("📈 Correlation Heatmap")

# Convert categorical features to numeric if needed (e.g., one-hot encode Hospital)
df_encoded = df.copy()
df_encoded["Hospital"] = df_encoded["Hospital"].astype("category").cat.codes

# Select only relevant numeric columns
corr_columns = ["Length of Stay", "Billing Amount", "Hospital"]  # "Hospital" is numeric now
corr = df_encoded[corr_columns].corr()

# Create and display the heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
st.pyplot(fig)

st.subheader("📊 Interactive Plot Explorer")

# Let user choose plot type and feature
plot_types = ["Histogram", "Boxplot", "Barplot"]
numerical_features = ["Age", "Billing Amount", "Length of Stay"]
categorical_features = ["Hospital", "Admission Type", "Insurance Provider"]

# Layout: two columns side-by-side
col1, col2 = st.columns([1, 2])

# Column 1: Controls
with col1:
    st.markdown("### Plot Controls")
    selected_plot = st.selectbox("Select Plot Type", plot_types)
    x_feature = st.selectbox("X-axis Feature", categorical_features)
    y_feature = st.selectbox("Y-axis Feature", numerical_features)

# Column 2: Plot output
with col2:
    fig, ax = plt.subplots(figsize=(7, 4))

    if selected_plot == "Histogram":
        sns.histplot(data=df, x=y_feature, kde=True, ax=ax)
        ax.set_title(f"Histogram of {y_feature}")

    elif selected_plot == "Boxplot":
        sns.boxplot(data=df, x=x_feature, y=y_feature, ax=ax)
        ax.set_title(f"{y_feature} by {x_feature}")

    elif selected_plot == "Barplot":
        sns.barplot(data=df, x=x_feature, y=y_feature, ax=ax, ci=None)
        ax.set_title(f"Average {y_feature} by {x_feature}")

    st.pyplot(fig)

import io

st.markdown("### Download Summary Report")

# Build your summary text
summary_text = f"""
Hospital Cost Prediction Report
===============================

Number of records: {len(df)}
Mean Billing Amount: ${df['Billing Amount'].mean():,.2f}
Mean Length of Stay: {df['Length of Stay'].mean():.1f} days

Top 3 Hospitals:
{df['Hospital'].value_counts().head(3).to_string(index=False)}

Generated using Streamlit
"""

# Convert text to bytes
report_bytes = summary_text.encode("utf-8")

# Streamlit download button (now using binary data)
st.download_button(
    label="📝 Download Text Report",
    data=report_bytes,
    file_name="summary_report.txt",
    mime="text/plain"
)
