import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title(" Customer Churn Prediction Dashboard")

# Load model safely
if not os.path.exists("model.pkl"):
    st.error(" Model file not found!")
    st.stop()

model = pickle.load(open("model.pkl", "rb"))

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):

    # Manual Encoding
    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": SeniorCitizen,
        "Partner": 1 if Partner == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if PhoneService == "Yes" else 0,
        "InternetService": 1 if InternetService == "Fiber optic" else 0,
        "Contract": 0 if Contract == "Month-to-month" else (1 if Contract == "One year" else 2),
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    input_df = pd.DataFrame([data])

    try:
        prediction = model.predict(input_df)

        # Check if model supports probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0]
        else:
            prob = [1 - prediction[0], prediction[0]]

        st.subheader("🔍 Prediction Result")

        if prediction[0] == 1:
            st.error(" Customer will churn")
        else:
            st.success(" Customer will not churn")

        #  Probability Graph
        st.subheader(" Churn Probability")

        fig, ax = plt.subplots()
        labels = ["No Churn", "Churn"]
        values = prob

        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

        st.pyplot(fig)

        # Input Visualization
        st.subheader(" Customer Profile")

        st.bar_chart(input_df.T)

    except Exception as e:
        st.error(f"Error: {e}")