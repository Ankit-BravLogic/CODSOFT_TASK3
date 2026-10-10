
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/churn_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a bank customer is likely to leave "
    "using a machine learning model."
)

st.divider()

st.subheader("Enter Customer Details")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900,
            value=650, step=1
        )
        geography = st.selectbox(
            "Geography", ["France", "Germany", "Spain"]
        )
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        age = st.number_input(
            "Age", min_value=18, max_value=100,
            value=35, step=1
        )
        tenure = st.number_input(
            "Tenure (years)", min_value=0, max_value=10,
            value=5, step=1
        )
        balance = st.number_input(
            "Account Balance", min_value=0.0,
            value=50000.0, step=1000.0
        )

    with col2:
        products = st.selectbox(
            "Number of Products", [1, 2, 3, 4]
        )
        has_card = st.selectbox(
            "Has Credit Card?", ["Yes", "No"]
        )
        active_member = st.selectbox(
            "Is Active Member?", ["Yes", "No"]
        )
        salary = st.number_input(
            "Estimated Salary", min_value=0.0,
            value=60000.0, step=1000.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

if submitted:
    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": salary
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to churn")
    else:
        st.success("✅ Prediction: Customer is likely to stay")

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    st.progress(float(probability))

    st.caption(
        "This is a machine learning estimate, not a guarantee "
        "of a customer's future behavior."
    )
