import streamlit as st
import pandas as pd
import plotly.express as px

# Load datasets
@st.cache_data
def load_data():
    return pd.read_excel("customer_agg_with_predictions.xlsx")

@st.cache_data
def load_transaction_data():
    return pd.read_excel("df.xlsx")

# Load datasets
data = load_data()
transaction_data = load_transaction_data()

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
    st.markdown(f"<h3 style='color:blue;'>Prediction: {prediction_text}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:green;'>Probability: {probability:.2f}%</h3>", unsafe_allow_html=True)

    # Display transaction-level data for the selected customer
    customer_transactions = transaction_data[transaction_data["CUSTOMERNAME"] == customer_name]

    if not customer_transactions.empty:
        st.subheader("Transaction-Level Insights")

        # Group transactions by year and month
        customer_transactions["Month"] = customer_transactions["PERIOD"]
        yearly_transactions = customer_transactions.groupby(["YEAR", "Month"])["Customer_Transactions"].sum().reset_index()

        # Create an interactive line plot
        fig = px.line(
            yearly_transactions,
            x="YEAR",
            y="Customer_Transactions",
            color="Month",
            markers=True,
            title=f"Yearly Transactions for {customer_name}",
            labels={"Customer_Transactions": "Transactions", "YEAR": "Year"},
        )
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig)

        # Display additional customer insights in a styled format
        st.subheader("Additional Customer Insights")
        insights = customer_transactions.iloc[0][[
            "Customer_Lifetime",
            "Months_Since_Last_Purchase",
            "Max_Time_Without_Purchase",
            "Trend_Classification",
            "Average_Purchase_Value",
        ]]
        st.markdown("<div style='display:flex; flex-wrap:wrap; gap:20px;'>", unsafe_allow_html=True)
        for key, value in insights.items():
            st.markdown(
                f"""
                <div style='background-color:#f8f9fa; border:1px solid #ddd; border-radius:8px; padding:15px; text-align:center; width:200px;'>
                    <h4 style='margin:0; color:#343a40;'>{key}</h4>
                    <p style='margin:0; font-size:20px; font-weight:bold; color:#007bff;'>{value}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"No transaction data available for {customer_name}.")
else:
    st.error("Customer not found in the dataset.")

