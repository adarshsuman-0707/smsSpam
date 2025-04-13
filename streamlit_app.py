# streamlit_app.py

import streamlit as st
import requests

st.title('📩 SMS Spam Classifier (Frontend)')

input_sms = st.text_input("Enter the Message")

if st.button("Predict"):
    if input_sms == "":
        st.warning("Please enter a message")
    else:
        try:
            res = requests.post("https://smsspam-a9ln.onrender.com/predict", json={"message": input_sms})
            result = res.json().get("prediction", "Error")
            st.success(f"Prediction: {result}")
        except:
            st.error("API not reachable. Make sure Flask backend is running.")
