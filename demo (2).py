import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your lab1 model (knn_model.joblib) instead of e_learning model
model = joblib.load('knn_model.joblib')

st.title("Course Recommendation System (course.csv style)")

# Input fields matching course.csv features
age = st.number_input("Age", min_value=16, max_value=80, value=25, step=1)
education = st.selectbox("Education", ["High School", "Undergraduate", "Graduate", "Working Professional"])
background = st.selectbox("Background", ["Science", "Arts", "Commerce", "Engineering"])
interest = st.selectbox("Interest", ["Finance", "Business", "Marketing", "Programming", "Cybersecurity", "Cloud"])
skill_level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
learning_style = st.selectbox("Learning Style", ["Reading", "Video", "Practical", "Project Based"])
study_time = st.selectbox("Study Time", ["<5", "5-10", "10-20", "20+"])
career_goal = st.selectbox("Career Goal", ["Skill Upgrade", "Get Job", "Start Business", "Freelancing", "Higher Studies"])
budget = st.selectbox("Budget", ["Free", "Low", "Medium", "Premium"])
duration = st.selectbox("Duration", ["1-3 months", "3-6 months", "6+ months"])

if st.button("Predict"):
    # Create a one-row DataFrame with the same feature names
    input_df = pd.DataFrame([{
        'Age': age,
        'Education': education,
        'Background': background,
        'Interest': interest,
        'SkillLevel': skill_level,
        'LearningStyle': learning_style,
        'StudyTime': study_time,
        'CareerGoal': career_goal,
        'Budget': budget,
        'Duration': duration
    }])

    # One-hot encode as in lab1
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns with zero (same as training features)
    # Load training schema from course.csv
    full_df = pd.read_csv('course.csv')
    X = full_df.drop('RecommendedCourse', axis=1)
    full_encoded = pd.get_dummies(X)

    for col in full_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Keep same order
    input_encoded = input_encoded[full_encoded.columns]

    y_label = full_df['RecommendedCourse']
    label_enc = LabelEncoder().fit(y_label)

    pred_encoded = model.predict(input_encoded)
    pred_label = label_enc.inverse_transform(pred_encoded)[0]

    st.success(f"Predicted course is: {pred_label}")
