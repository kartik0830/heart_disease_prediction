import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_disease_model.pkl")

st.title("❤️ Heart Disease Prediction App")

age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [1, 0])
restecg = st.selectbox("Resting ECG (0–2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [1, 0])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])
thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1,2,3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {prob:.2f})")
