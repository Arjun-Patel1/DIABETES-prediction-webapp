# app.py — Streamlit App
import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
#with open("../model/diabetes_model.pkl", "rb") as f:
 #   model, scaler = pickle.load(f)
with open("model/diabetes_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Streamlit App
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🔍 Diabetes Risk Prediction App")
st.write("Enter the following information to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    result = model.predict(scaled_data)

    if result[0] == 1:
        st.error("⚠️ Prediction: You may be at risk of Diabetes.")
    else:
        st.success("✅ Prediction: You are not likely to have Diabetes.")
