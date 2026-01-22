import streamlit as st
import joblib

model = joblib.load('knn_model.joblib')
st.title("Books Classifier")

Num_Pages = st.number_input(label="Num Pages",min_value=0.0,max_value=10.0,value=5.0)
Avg_WordLength= st.number_input(label="Avg WordLength",min_value=0.0,max_value=10.0,value=3.5)
Reading_TimeHours = st.number_input(label="Reading TimeHours",min_value=0.0,max_value=10.0,value=1.5)
Complexity_Score = st.number_input(label="Complexity Score",min_value=0.0,max_value=10.0,value=0.2)
sample=[[Num_Pages,Avg_WordLength ,Reading_TimeHours  , Complexity_Score]]

if st.button("Predict"):
    prediction = model.predict(sample)[0]
    st.success(f"Predicted species is {prediction}")