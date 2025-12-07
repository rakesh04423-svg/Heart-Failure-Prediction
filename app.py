import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing objects
cat_model = joblib.load("cat_model.pkl")
encoders = joblib.load("encoders.pkl")
scalers = joblib.load("scalers.pkl")

st.title("❤️ Heart Disease Prediction App")


# Collect user input (only training features!)
age = st.number_input("Age", min_value=18, max_value=100, value=30) 
sex = st.radio("Sex", ["M", "F"]) 
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"]) 
#resting_bp = st.number_input("RestingBP", min_value=80, max_value=200, value=120) 
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200) 
fasting_bs = st.radio("FastingBS", [0, 1]) 
#resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"]) 
max_hr = st.number_input("MaxHR", min_value=60, max_value=220, value=150) 
exercise_angina = st.radio("ExerciseAngina", ["Y", "N"]) 
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0) 
st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])

# Put inputs into a dataframe
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [chest_pain],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "MaxHR": [max_hr],
    "ExerciseAngina": [exercise_angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [st_slope]
})

# Apply encoders
for col in encoders:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

# Apply scalers
for col in scalers:
    if col in input_df.columns:
        input_df[col] = scalers[col].transform(input_df[[col]])

# Predict
if st.button("Predict"):
    prediction = cat_model.predict(input_df.values)
    if prediction[0] == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")