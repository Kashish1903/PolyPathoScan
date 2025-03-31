
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import tensorflow.keras.backend as K


# Load models
malaria_model = tf.keras.models.load_model("/content/malariaModel (1).h5")
heart_model = joblib.load("/content/heart_disease_model.sav")
diabetes_model = joblib.load("/content/diabetes_model.sav")
pneumonia_model = tf.keras.models.load_model("/content/model.h5")  

IMG_SIZE = 150  # Malaria model expects 150x150 images

# Function to preprocess image for Malaria model
def preprocess_image(image, target_size=(150, 150)):
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize properly
    img_array = img_array / 255.0  # Normalize using actual max pixel value
    print("Preprocessed Image Shape:", img_array.shape)  # Debugging
    print("Min Pixel Value:", img_array.min(), "Max Pixel Value:", img_array.max())  # Debugging
    return img_array
      # Apply scaling

      # Clip values to prevent extreme outputs
    prediction = np.clip(prediction, 0.01, 0.99)

def adjust_predictions(raw_prediction, temperature=2.0):
    # Apply temperature scaling to adjust model confidence
    scaled_prediction = raw_prediction / temperature
    adjusted_pred = K.sigmoid(scaled_prediction)  # Ensure values are well-scaled
    return adjusted_pred.numpy()

# Function to preprocess image for Pneumonia model
def predict_pneumonia(image):
    img = Image.open(image).resize((128, 128)).convert("RGB")  # Load and resize image
    img_array = np.array(img) / 255.0  # Convert to NumPy array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = pneumonia_model.predict(img_array)  # Make prediction
    return prediction[0][0]  # Return the raw prediction value

# Heart Disease Input Fields
def heart_disease_form():
    cols = st.columns(3)
    fields = ["Age", "Resting Blood Pressure", "Fasting Blood Sugar", "Sex", "Serum Cholesterol", "Resting ECG", "Chest Pain Type", "Max Heart Rate", "Exercise-Induced Angina"]
    inputs = []
    for i, field in enumerate(fields):
        with cols[i % 3]:
            inputs.append(st.number_input(field))
    return inputs

# Diabetes Input Fields
def diabetes_form():
    cols = st.columns(3)
    fields = ["Pregnancies", "Blood Pressure", "Insulin", "Glucose Level", "Skin Thickness", "BMI", "Diabetes Pedigree Function", "Age"]
    inputs = []
    for i, field in enumerate(fields):
        with cols[i % 3]:
            inputs.append(st.number_input(field))
    return inputs

# Streamlit App
st.title("PolyPathoScan")
option = st.sidebar.selectbox("Select Disease", ["Malaria", "Pneumonia", "Heart Disease", "Diabetes"])


if option == "Malaria":
    st.header("Malaria Prediction")
    uploaded_file = st.file_uploader("Upload Cell Image", type=["jpg", "png"])
    if uploaded_file is not None:
        img_array = preprocess_image(uploaded_file)
        raw_prediction = malaria_model.predict(img_array)
        print("Raw Malaria Model Output:", raw_prediction)  # Debugging
        prediction = adjust_predictions(raw_prediction)
        print("Adjusted Malaria Model Output:", prediction)  # Debugging
        st.write("Positive " if prediction[0][0] > 0.5 else "Negative ✅")


elif option == "Pneumonia":
    st.header("Pneumonia Prediction")
    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png"])
    if uploaded_file is not None:
        prediction = predict_pneumonia(uploaded_file)  # Get prediction value
        st.write("Positive " if prediction > 0.5 else "Negative ✅")

elif option == "Heart Disease":
    st.header("Heart Disease Prediction")
    user_input = heart_disease_form()
    if st.button("Predict Heart Disease"):
        prediction = heart_model.predict([user_input])
        print("Raw Heart Model Output:", prediction)  # Debugging
        st.write("Positive " if prediction[0] == 1 else "Negative ✅")

elif option == "Diabetes":
    st.header("Diabetes Prediction")
    user_input = diabetes_form()
    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict([user_input])
        print("Raw Diabetes Model Output:", prediction)  # Debugging
        st.write("Positive " if prediction[0] == 1 else "Negative ✅")

