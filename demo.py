import streamlit as st
import joblib

model = joblib.load('knn_model.joblib')
st.title("Iris File Classifier")

sepal_length = st.number_input(label="Sepal Length",min_value=0.0,max_value=10.0,value=5.0)
sepal_width = st.number_input(label="Sepal width",min_value=0.0,max_value=10.0,value=3.5)
petal_length = st.number_input(label="Petal Length",min_value=0.0,max_value=10.0,value=1.5)
petal_width = st.number_input(label="Petal Width",min_value=0.0,max_value=10.0,value=0.2)
sample=[[sepal_length, sepal_width, petal_length, petal_width]]

if st.button("Predict"):
    prediction = model.predict(sample)[0]
    st.success(f"Predicted species is {prediction}")