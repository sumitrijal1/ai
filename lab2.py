import streamlit as st
import joblib

st.title ("News categroy prediction")
input_text =st.text_input("enter the news you want to predict")

model = joblib.load("navive_bayes.joblib")
if st.button("PREDICT"):
    output = model.predict([input_text])
    st.success(output[0])