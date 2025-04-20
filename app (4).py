# ---------------------- PolyPathoScan App ----------------------

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import csv
import uuid
import os
import requests
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from fpdf import FPDF

# ==== White Theme Custom CSS ====
def local_css():
    st.markdown(
        """
        <style>
        body {background-color: #ffffff;}
        .css-18e3th9 {background-color: #ffffff;}
        header, footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """, unsafe_allow_html=True
    )
local_css()

# ==== Model Loading ====
MALARIA_MODEL_PATH = "malaria_model.h5"
MALARIA_MODEL_URL = "https://drive.google.com/uc?id=1DSTFSf9YhuV0_wg7NIP93x-gg2M4lL5u"


if not os.path.exists(MALARIA_MODEL_PATH):
    with open(MALARIA_MODEL_PATH, "wb") as f:
        r = requests.get(MALARIA_MODEL_URL)
        f.write(r.content)

malaria_model = tf.keras.models.load_model("malaria_model_small.h5")
heart_model = joblib.load("heart_model.sav")
diabetes_model = joblib.load("diabetes_model.sav")
pneumonia_model = tf.keras.models.load_model("pneumonia_model.h5")

# ==== Patient Storage CSV ====
PATIENT_CSV = "patients.csv"
if not os.path.exists(PATIENT_CSV):
    with open(PATIENT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Age", "Gender", "Mobile", "Disease", "Result"])

# ==== Authentication ====
credentials = {
    "usernames": {
        "admin": {"name": "Admin", "password": "$2b$12$Yu92MfNKz4qDKDxeRgHWze5ZZefgv42/K7e4WhSqBcb.nWQIeYSSa"},
        "doctor": {"name": "Doctor", "password": "$2b$12$m.XqFJzdFXpeMco42yGchOwXilqFcmD9tOKfjrVpoNZjVepmIY8aO"},
        "viewer": {"name": "Viewer", "password": "$2b$12$Is.879loXJ4Bm1n5WyyTtedxIHW8IoWccTSjgop90acD6mmOaz4zm"},
    }
}
authenticator = stauth.Authenticate(credentials, "cookie", "auth", 1)
name, auth_status, username = authenticator.login("Login Form", location="main")


# ==== Sidebar ====
with st.sidebar:
    st.image("hospital-logo-clinic-health-care-physician-business-removebg-preview.png", width=50)
    st.title("PolyPathoScan ")

# ==== PDF Report Generator ====
def generate_pdf(patient_info, disease, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "PolyPathoScan - Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"{disease} Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Patient Details:", ln=True)
    for key, value in patient_info.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    pdf.ln(10)

    # Clean Result
    if "Positive" in result:
        clean_result = "Positive"
    elif "Negative" in result:
        clean_result = "Negative"
    else:
        clean_result = result

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Prediction Result: {clean_result}", ln=True)
    pdf.ln(5)

    if disease in ["Malaria", "Pneumonia"]:
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, "Precautionary Advice:\n- Stay Hydrated\n- Complete Medication\n- Rest Properly\n- Seek Medical Consultation")
    else:
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, "Conclusions:\n- Maintain Healthy BMI\n- Monitor Blood Pressure\n- Regular Checkups Recommended")

    pdf.ln(15)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This report is generated automatically using Machine Learning Models and should not replace professional medical advice. Please consult a qualified doctor for confirmation.")

    pdf.ln(20)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Doctor's Signature: ____________________", ln=True, align='R')

    filename = f"{patient_info['Name']}_Report.pdf"
    pdf.output(filename)
    return filename

# ==== Main App ====
if auth_status:
    authenticator.logout("Logout", "sidebar")
    role_map = {"admin": "admin", "doctor": "doctor", "viewer": "viewer"}
    role = role_map.get(username, "unknown")

    if role in ["admin", "doctor"]:
        page = st.sidebar.selectbox("Select Page", ["Patient Testing", "Admin Dashboard" if role == "admin" else ""])
    elif role == "viewer":
        page = "View Patients"

    if page == "Patient Testing":
        st.title("Patient Registration")
        with st.form("patient_form"):
            col1, col2, col3 = st.columns(3)
            pname = col1.text_input("Name")
            page_age = col2.number_input("Age", min_value=0, step=1)
            psex = col3.selectbox("Gender", ["Male", "Female"])
            pmobile = st.text_input("Mobile Number")
            submitted = st.form_submit_button("Proceed to Testing")

        if submitted and pname and pmobile:
            st.session_state.current_patient = {
                "id": str(uuid.uuid4())[:8],
                "name": pname,
                "age": page_age,
                "gender": psex,
                "mobile": pmobile
            }

        if "current_patient" in st.session_state:
            st.header("Select Disease to Test")
            option = st.selectbox("Choose Disease", ["Malaria", "Pneumonia", "Heart Disease", "Diabetes"])

            result = None

            if option == "Malaria":
                uploaded_file = st.file_uploader("Upload Cell Image", type=["jpg", "png"])
                if uploaded_file:
                    with st.spinner("Analyzing Image..."):
                        img_array = load_img(uploaded_file, target_size=(150, 150))
                        img_array = img_to_array(img_array)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0
                        raw_prediction = malaria_model.predict(img_array)
                        prediction = K.sigmoid(raw_prediction/2.0).numpy()
                        result = "Positive" if prediction[0][0] > 0.5 else "Negative"
                        st.success(f"Result: {result}")
                        st.session_state.prediction_result = result
                    with open(PATIENT_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                        st.session_state.current_patient.get("id", ""),
                        st.session_state.current_patient.get("name", ""),
                        st.session_state.current_patient.get("age", ""),
                        st.session_state.current_patient.get("gender", ""),
                        st.session_state.current_patient.get("mobile", ""),
                        option,
                        st.session_state.prediction_result
                        ])


            elif option == "Pneumonia":
                uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png"])
                if uploaded_file:
                    with st.spinner("Analyzing X-ray..."):
                        img = Image.open(uploaded_file).resize((128, 128)).convert("RGB")
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        prediction = pneumonia_model.predict(img_array)
                        result = "Positive" if prediction[0][0] > 0.5 else "Negative"
                        st.success(f"Result: {result}")
                        st.session_state.prediction_result = result
                    with open(PATIENT_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                        st.session_state.current_patient.get("id", ""),
                        st.session_state.current_patient.get("name", ""),
                        st.session_state.current_patient.get("age", ""),
                        st.session_state.current_patient.get("gender", ""),
                        st.session_state.current_patient.get("mobile", ""),
                        option,
                        st.session_state.prediction_result
                        ])



            elif option == "Heart Disease":
                fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
                cols = st.columns(3)
                values = [cols[idx % 3].number_input(field.capitalize(), step=1.0) for idx, field in enumerate(fields)]
                if st.button("Predict Heart Disease"):
                    with st.spinner("Predicting..."):
                        prediction = heart_model.predict([values])
                        result = "Positive" if prediction[0] == 1 else "Negative"
                        st.success(f"Result: {result}")
                        st.session_state.prediction_result = result
                    with open(PATIENT_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                        st.session_state.current_patient.get("id", ""),
                        st.session_state.current_patient.get("name", ""),
                        st.session_state.current_patient.get("age", ""),
                        st.session_state.current_patient.get("gender", ""),
                        st.session_state.current_patient.get("mobile", ""),
                        option,
                        st.session_state.prediction_result
                        ])


            elif option == "Diabetes":
                fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
                cols = st.columns(3)
                values = [cols[idx % 3].number_input(field, step=1.0) for idx, field in enumerate(fields)]
                if st.button("Predict Diabetes"):
                    with st.spinner("Predicting..."):
                        prediction = diabetes_model.predict([values])
                        result = "Positive" if prediction[0] == 1 else "Negative"
                        st.success(f"Result: {result}")
                        st.session_state.prediction_result = result
                    with open(PATIENT_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                        st.session_state.current_patient.get("id", ""),
                        st.session_state.current_patient.get("name", ""),
                        st.session_state.current_patient.get("age", ""),
                        st.session_state.current_patient.get("gender", ""),
                        st.session_state.current_patient.get("mobile", ""),
                        option,
                        st.session_state.prediction_result
                        ])


            if "prediction_result" in st.session_state:
                patient_info = {
                    "Name": st.session_state.current_patient.get("name", "NA"),
                    "Age": st.session_state.current_patient.get("age", "NA"),
                    "Gender": st.session_state.current_patient.get("gender", "NA"),
                    "Mobile Number": st.session_state.current_patient.get("mobile", "NA")
                }
                if st.button("Generate Report 📄"):
                    report_file = generate_pdf(patient_info, option, st.session_state.get("prediction_result", "Result Not Available"))
                    with open(report_file, "rb") as f:
                        st.download_button("Download Report", f, file_name=report_file, mime="application/pdf")

    elif page == "Admin Dashboard" and role == "admin":
        st.title("📊 Admin Dashboard")
        df = pd.read_csv(PATIENT_CSV)
        st.metric("Total Patients Tested", len(df))
        disease_counts = df["Disease"].value_counts()
        st.subheader("Disease Wise Distribution")
        st.bar_chart(disease_counts)
        st.subheader("Gender Distribution")
        if "Gender" in df.columns:
            gender_counts = df["Gender"].value_counts()
        elif "Sex" in df.columns:
            gender_counts = df["Sex"].value_counts()
        else:
            gender_counts = pd.Series()
        if not gender_counts.empty:
            fig, ax = plt.subplots()
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)
        else:
            st.info("No gender data available.")
        st.subheader("Most Common Disease")
        if not disease_counts.empty:
            st.success(f" {disease_counts.idxmax()}")
        st.subheader("📝 Full Patient Reports")
        st.dataframe(df)
        st.download_button("Download Patient Reports", df.to_csv(index=False), file_name="patients_full.csv")

    elif page == "View Patients":
        st.title("📝 View Patient Reports")
        df = pd.read_csv(PATIENT_CSV)
        st.dataframe(df)
        st.download_button("Download All Patients CSV", df.to_csv(index=False), file_name="patients_list.csv")

elif auth_status is False:
    st.error(" Incorrect username or password")
elif auth_status is None:
    st.warning(" Please enter your login credentials")
