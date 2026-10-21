from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "churn_model.pkl")
    preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model()

FIELDS = {
    "Customer profile": {
        "Gender": ("gender", ["Male", "Female"]),
        "Senior Citizen": ("SeniorCitizen", [0, 1]),
        "Partner": ("Partner", ["Yes", "No"]),
        "Dependents": ("Dependents", ["Yes", "No"]),
        "Tenure (months)": ("tenure", (0.0, 100.0, 1.0)),
    },
    "Services and billing": {
        "Phone Service": ("PhoneService", ["Yes", "No"]),
        "Multiple Lines": ("MultipleLines", ["Yes", "No", "No phone service"]),
        "Internet Service": ("InternetService", ["DSL", "Fiber optic", "No"]),
        "Online Security": ("OnlineSecurity", ["Yes", "No", "No internet service"]),
        "Online Backup": ("OnlineBackup", ["Yes", "No", "No internet service"]),
        "Device Protection": ("DeviceProtection", ["Yes", "No", "No internet service"]),
        "Tech Support": ("TechSupport", ["Yes", "No", "No internet service"]),
        "Streaming TV": ("StreamingTV", ["Yes", "No", "No internet service"]),
        "Streaming Movies": ("StreamingMovies", ["Yes", "No", "No internet service"]),
        "Contract": ("Contract", ["Month-to-month", "One year", "Two year"]),
        "Paperless Billing": ("PaperlessBilling", ["Yes", "No"]),
        "Payment Method": ("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
        "Monthly Charges": ("MonthlyCharges", (0.0, None, 0.01)),
        "Total Charges": ("TotalCharges", (0.0, None, 0.01)),
    },
}

st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate whether the customer is likely to churn.")

with st.form("customer_details"):
    columns = st.columns(len(FIELDS))
    customer = {}

    for column, (section, fields) in zip(columns, FIELDS.items()):
        with column:
            st.subheader(section)
            for label, (name, options) in fields.items():
                if isinstance(options, tuple):
                    minimum, maximum, step = options
                    customer[name] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        step=step,
                    )
                elif name == "SeniorCitizen":
                    customer[name] = st.selectbox(
                        label,
                        options,
                        format_func=lambda value: "Yes" if value else "No",
                    )
                else:
                    customer[name] = st.selectbox(label, options)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer_frame = pd.DataFrame([customer])
    prediction = model.predict(preprocessor.transform(customer_frame))[0]

    if prediction == "Yes":
        st.warning("Prediction: This customer is likely to churn.")
    else:
        st.success("Prediction: This customer is not likely to churn.")
