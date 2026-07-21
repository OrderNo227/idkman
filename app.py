

import joblib
import numpy as np
import streamlit as st

# App title
st.title("Machine Learning Model Predictor")

# Load your trained model
model = joblib.load("model.pkl")

# Add input fields (Customize these based on your model's inputs)
st.write("### Enter Feature Values:")
input_1 = st.number_input("Feature 1", value=0.0)
input_2 = st.number_input("Feature 2", value=0.0)

# Prediction logic
if st.button("Predict"):
    features = np.array([[input_1, input_2]])
    prediction = model.predict(features)
    st.success(f"Result: {prediction[0]}")