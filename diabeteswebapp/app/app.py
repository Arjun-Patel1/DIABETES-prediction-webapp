import streamlit as st
import numpy as np
import pickle

# ----- Dummy Transformation Function -----
def transform_input(input_array):
    """
    Transforms the raw input of shape (1, 8) into shape (1, 18)
    by appending 10 dummy features (zeros).

    Replace this logic with your actual preprocessing pipeline.
    """
    # Number of extra features needed:
    extra_feature_count = 18 - input_array.shape[1]
    extra_features = np.zeros((input_array.shape[0], extra_feature_count))
    transformed = np.hstack((input_array, extra_features))
    return transformed

# ----- Load the Model -----
# Here we load the model from the pickle file stored in the relative path "model/diabetes_model.pkl"
with open("model/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----- App Configuration and Custom Styling (Optional) -----
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🔍 Diabetes Risk Prediction App")
st.write("Enter the following information to predict the likelihood of diabetes.")

# ----- Input Fields -----
# We collect the 8 raw features from the user.
pregnancies    = st.number_input("Pregnancies", min_value=0, step=1)
glucose        = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin        = st.number_input("Insulin Level", min_value=0)
bmi            = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf            = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age            = st.number_input("Age", min_value=1)

# ----- Prediction -----
if st.button("Predict"):
    # Create a raw input array (shape will be (1,8))
    raw_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age]])
    
    # Transform the raw input to match the expected 18 features
    input_data = transform_input(raw_input)
    
    # Now predict using the model
    prediction = model.predict(input_data)
    
    # Display result based on the prediction outcome
    if prediction[0] == 1:
        st.error("⚠️ Prediction: You may be at risk of Diabetes.")
    else:
        st.success("✅ Prediction: You are not likely to have Diabetes.")
