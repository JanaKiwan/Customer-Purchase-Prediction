import streamlit as st
import pandas as pd

# Load the Excel file
@st.cache_data
def load_data():
    return pd.read_excel("customer_agg_with_predictions.xlsx")

# Load the dataset
data = load_data()

# Map metrics and thresholds for each model
model_metrics = {
    "Lasso Logistic Regression": {
        "threshold": 0.31,
        "roc_auc": 0.93,
        "true_positive_rate": "91%",
        "false_negative_rate": "15%",
    },
    "L2 Logistic Regression": {
        "threshold": 0.31,
        "roc_auc": 0.93,
        "true_positive_rate": "91%",
        "false_negative_rate": "15%",
    },
    "Calibrated SVC": {
        "threshold": 0.35,
        "roc_auc": 0.93,
        "true_positive_rate": "90%",
        "false_negative_rate": "21%",
    },
    "Decision Tree": {
        "threshold": 0.39,
        "roc_auc": 0.90,
        "true_positive_rate": "84%",
        "false_negative_rate": "9%",
    },
}

# Model columns
model_columns = {
    "Lasso Logistic Regression": (
        "Lasso Logistic Regression_Prediction",
        "Lasso Logistic Regression_Probability",
    ),
    "L2 Logistic Regression": (
        "L2 Logistic Regression_Prediction",
        "L2 Logistic Regression_Probability",
    ),
    "Calibrated SVC": (
        "Calibrated SVC_Prediction",
        "Calibrated SVC_Probability",
    ),
    "Decision Tree": (
        "Decision Tree_Prediction",
        "Decision Tree_Probability",
    ),
}

# Streamlit App
st.title("Customer Purchase Prediction Viewer")

# Model Selection
selected_model = st.selectbox("Select a Model", list(model_columns.keys()))

# Sidebar Metrics
st.sidebar.header("Model Metrics")
st.sidebar.metric("Threshold", model_metrics[selected_model]["threshold"])
st.sidebar.metric("ROC AUC Score", model_metrics[selected_model]["roc_auc"])
st.sidebar.metric("True Positive Rate", model_metrics[selected_model]["true_positive_rate"])
st.sidebar.metric("False Negative Rate", model_metrics[selected_model]["false_negative_rate"])

# Customer Dropdown
customer_name = st.selectbox("Select a Customer", data["CUSTOMERNAME"].unique())

# Extract model-specific columns
prediction_col, probability_col = model_columns[selected_model]

# Filter data for the selected customer
customer_data = data[data["CUSTOMERNAME"] == customer_name]

if not customer_data.empty:
    st.subheader(f"Prediction Details for {customer_name}")
    prediction = customer_data[prediction_col].iloc[0]
    probability = customer_data[probability_col].iloc[0] * 100  # Convert to percentage

    # Display outcomes
    prediction_text = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.write(f"**Prediction:** {prediction_text}")
    st.write(f"**Probability:** {probability:.2f}%")
else:
    st.error("Customer not found in the dataset.")

