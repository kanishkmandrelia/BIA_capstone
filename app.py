# ["Years_At_Company", "Monthly_Salary", "Overtime_Hours", "Promotions", "Employee_Satisfaction_Score"]

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("svc_model.pkl")

st.title("Employee Performance Score Prediction")

st.divider()

st.write("You can get the performance estimation of the employee after entering the values and pressing to the predict button")

st.divider()

year = st.number_input("Enter the years of company", min_value=0, max_value=15, value=2)
salary = st.number_input("Enter the monthly salary of the employee", min_value=500.0, max_value=50000.0, value=4000.0)
overtime = st.number_input("Enter the Overtime hours", min_value=0, max_value=90, value=10)
promotion = st.number_input("Enter the number of promotions", min_value=0, max_value=10, value=1)
satisfaction = st.number_input("enter employee satisfaction", min_value=0.0, max_value=5.0, value=2.0)

x = [year, salary, overtime, promotion, satisfaction]

st.divider()

predict_button = st.button("Predict the employee performance score!")

st.divider()

if predict_button:
    x_array = np.array(x)

    x_scale = scaler.transform([x_array])
    predict = model.predict(x_scale)[0]

    st.snow()
    st.balloons()

    st.write(f"Prediction for the performance score is: {predict}")

else:
    st.write("Please use the button for the prediction")
