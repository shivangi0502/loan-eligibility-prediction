import streamlit as st
import pickle
import os

# Load the trained model
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
model_file_path = os.path.join(dataset_path, 'logistic_model.pkl')

with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

st.title("Loan Prediction App")

# Collect user-friendly input
st.header("Enter Loan Details")
credit_history = st.selectbox("Credit History", ["No Credit History", "Has Credit History"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
gender = st.selectbox("Gender", ["Female", "Male"])

# Convert inputs to encoded values
credit_history_encoded = 1 if credit_history == "Has Credit History" else 0
education_encoded = 1 if education == "Not Graduate" else 0
gender_encoded = 1 if gender == "Male" else 0

# Predict function
def predict(credit_history, education, gender):
    input_data = [[credit_history, education, gender]]
    prediction = model.predict(input_data)
    return "Approved" if prediction[0] == 1 else "Not Approved"

# Display prediction
if st.button("Predict Loan Status"):
    result = predict(credit_history_encoded, education_encoded, gender_encoded)
    st.write(f"Loan Status: {result}")
