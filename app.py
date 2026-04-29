import streamlit as st
import joblib
import numpy as np

# Load models
clf = joblib.load("classifier.pkl")
reg = joblib.load("regressor.pkl")

st.title("🏠 Real Estate Investment Predictor")

# Inputs
bhk = st.number_input("BHK")
area = st.number_input("Area (sqft)")
price = st.number_input("Current Price")

if st.button("Predict"):
    input_data = np.array([[bhk, area, price]])

    invest = clf.predict(input_data)
    future_price = reg.predict(input_data)

    st.subheader("Result:")
    st.write("Good Investment:", "Yes" if invest[0]==1 else "No")
    st.write("Future Price:", future_price[0])