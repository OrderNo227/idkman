import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

st.set_page_config(page_title='Health Status Predictor', layout='centered')

st.title('Smart Healthcare - Health Status Predictor')
st.write('Enter the patient details below to predict their health status.')

# Input fields for features
st.sidebar.header('Patient Input Features')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age (years)', 18, 90, 30)
    height = st.sidebar.slider('Height (meter)', 1.0, 2.5, 1.75)
    weight = st.sidebar.slider('Weight (kg)', 30.0, 150.0, 70.0)
    bmi = st.sidebar.slider('BMI', 15.0, 40.0, 22.0)
    step_count = st.sidebar.slider('Step Count', 0, 20000, 5000)
    distance_travel = st.sidebar.slider('Distance Travel (Km)', 0.0, 20.0, 5.0)
    heart_rate = st.sidebar.slider('Heart Rate (BPM)', 40, 180, 70)
    blood_oxygen_level = st.sidebar.slider('Blood Oxygen Level', 80.0, 100.0, 98.0)
    sleep_duration = st.sidebar.slider('Sleep Duration (minutes)', 120, 720, 480)
    screen_time = st.sidebar.slider('Screen Time (minute)', 0, 720, 180)
    earphone_time = st.sidebar.slider('Earphone Time (minute)', 0, 720, 60)
    systolic_bp = st.sidebar.slider('Systolic Blood Pressure', 80, 200, 120)
    diastolic_bp = st.sidebar.slider('Diastolic Blood Pressure', 40, 120, 80)

    data = {
        'Age (years)': age,
        'Height (meter)': height,
        'Weight (kg)': weight,
        'BMI': bmi,
        'Step Count': step_count,
        'Distance Travel (Km)': distance_travel,
        'Heart Rate (BPM)': heart_rate,
        'Blood Oxygen Level': blood_oxygen_level,
        'Sleep Duration (minutes)': sleep_duration,
        'Screen Time (minute)': screen_time,
        'Earphone Time (minute)': earphone_time,
        'Systolic Blood Pressure': systolic_bp,
        'Diastolic Blood Pressure': diastolic_bp,
        'Gender_Male': 1 if gender == 'Male' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input:')
st.write(df_input)

if st.button('Predict Health Status'):
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.subheader('Prediction:')
    if prediction[0] == True:
        st.error('The predicted health status is: **UNHEALTHY**')
    else:
        st.success('The predicted health status is: **HEALTHY**')

    st.subheader('Prediction Probability:')
    st.write(f"Healthy: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Unhealthy: {prediction_proba[0][1]*100:.2f}%")

st.markdown("""
--- 
This application uses a Logistic Regression model to predict health status based on various lifestyle and physiological metrics. 'Unhealthy' status is defined based on statistical anomalies in key health metrics using the Interquartile Range (IQR) method.
""")
