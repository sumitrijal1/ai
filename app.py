from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('knn_model.joblib')

# Load the original data to understand the feature structure
df = pd.read_csv('course.csv')

# Prepare the preprocessing (same as in lab1.ipynb)
X = df.drop('RecommendedCourse', axis=1)
y = df['RecommendedCourse']

# One-hot encode features (same as training)
X_encoded = pd.get_dummies(X)

# Fit label encoder on target (same as training)
le = LabelEncoder()
le.fit(y)

# Store feature names for consistent preprocessing
feature_names = X_encoded.columns.tolist()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Create input DataFrame with the same structure
        input_data = pd.DataFrame([{
            'Age': int(data['age']),
            'Education': data['education'],
            'Background': data['background'],
            'Interest': data['interest'],
            'SkillLevel': data['skillLevel'],
            'LearningStyle': data['learningStyle'],
            'StudyTime': data['studyTime'],
            'CareerGoal': data['careerGoal'],
            'Budget': data['budget'],
            'Duration': data['duration']
        }])

        # One-hot encode the input data (same as training)
        input_encoded = pd.get_dummies(input_data)

        # Ensure all expected columns are present (add missing columns with 0)
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Ensure columns are in the same order as training
        input_encoded = input_encoded[feature_names]

        # Make prediction
        prediction = model.predict(input_encoded)

        # Convert back to original course name
        recommended_course = le.inverse_transform(prediction)[0]

        return jsonify({
            'success': True,
            'recommended_course': recommended_course
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/')
def home():
    return "Course Recommendation API is running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)