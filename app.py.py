import streamlit as st
import joblib
import pandas as pd

st.title("Supervised ML Model")

st.write("Enter input values")

age = st.number_input("Age")
hours = st.number_input("Hours per week")

if st.button("Predict"):
    
    model = joblib.load("model.pkl")
    
    prediction = model.predict([[age, hours]])
    
    st.write("Prediction:", prediction)