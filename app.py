# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('titanic_model.pkl')
le_sex = joblib.load('le_sex.pkl')
le_embarked = joblib.load('le_embarked.pkl')

st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")

# App Header
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Predict whether a passenger would have survived the Titanic disaster using ML.")

st.sidebar.header("🧾 Input Passenger Details")

# Input Fields
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid", 0, 500, 50)
embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode inputs
sex_encoded = le_sex.transform([sex])[0]
embarked_encoded = le_embarked.transform([embarked])[0]

# Combine features
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 The passenger would have **SURVIVED** with a probability of {prob:.2%}")
    else:
        st.error(f"☠️ Unfortunately, the passenger would **NOT SURVIVE**. Survival probability: {prob:.2%}")
    
    st.markdown("---")
    st.subheader("🔍 Prediction Details")
    st.json({
        "Passenger Class": pclass,
        "Sex": sex,
        "Age": age,
        "Siblings/Spouses": sibsp,
        "Parents/Children": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Predicted": "Survived" if prediction == 1 else "Not Survived",
        "Probability of Survival": f"{prob:.2%}"
    })
