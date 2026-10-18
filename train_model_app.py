#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("model.pkl")

st.set_page_config(page_title="Diabetes Classification", page_icon="📊")

st.title("📊 Diabetes Classification")

st.write("Logistic Regression model demonstration")

# User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0.0, format="%.4f")
glucose  = st.number_input("Glucose ", min_value=0.0, format="%.4f")
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, format="%.4f")
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, format="%.4f")
insulin = st.number_input("Insulin", min_value=0.0, format="%.4f")
bmi = st.number_input("BMI", min_value=0.0, format="%.4f")
pedigree  = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.4f")
age = st.number_input("Age", min_value=0.0, format="%.4f")

if st.button("Predict Diagnosis"):

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    # Match the training feature order
    input_data = input_data[features]

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]


    # Display Result
    st.subheader("Model Output")
    st.write("Predicted class:", int(prediction))
    st.write(
        "Estimated probability for class 1:",
        f"{probability:.2%}"
    )
