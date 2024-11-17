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
    st.write(f"**Prediction:** {prediction_text}")
    st.write(f"**Probability:** {probability:.2f}%")

    # Display transaction-level data for the selected customer
    customer_transactions = transaction_data[transaction_data["CUSTOMERNAME"] == customer_name]

    if not customer_transactions.empty:
        st.subheader("Transaction-Level Insights")

        # Aggregate transactions by year and month
        customer_transactions["Month"] = customer_transactions["PERIOD"]
        yearly_transactions = (
            customer_transactions.groupby(["YEAR", "Month"])["Customer_Transactions"]
            .sum()
            .reset_index()
        )

        # Create an interactive line plot using Plotly
        fig = px.line(
            yearly_transactions,
            x="YEAR",
            y="Customer_Transactions",
            color_discrete_sequence=["#636EFA"],
            markers=True,
            title=f"Yearly Transactions for {customer_name}",
        )
        fig.update_traces(hovertemplate="Year: %{x}<br>Month: %{customdata}<br>Transactions: %{y}")
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Transactions",
            xaxis=dict(tickmode="array", tickvals=yearly_transactions["YEAR"].unique()),
            template="plotly_white",
        )
        fig.update_traces(customdata=yearly_transactions["Month"])
        st.plotly_chart(fig, use_container_width=True)

        # Display additional customer insights in a horizontal format
        st.markdown("---")
        st.subheader("Additional Customer Insights")
        cols = st.columns(5)

        cols[0].metric("Customer Lifetime", customer_transactions["Customer_Lifetime"].iloc[0])
        cols[1].metric(
            "Months Since Last Purchase",
            customer_transactions["Months_Since_Last_Purchase"].iloc[0],
        )
        cols[2].metric(
            "Max Time Without Purchase",
            customer_transactions["Max_Time_Without_Purchase"].iloc[0],
        )
        cols[3].metric(
            "Trend Classification",
            customer_transactions["Trend_Classification"].iloc[0],
        )
        cols[4].metric(
            "Average Purchase Value (AED)",
            f"{customer_transactions['Average_Purchase_Value'].iloc[0]:,.2f}",
        )
    else:
        st.warning(f"No transaction data available for {customer_name}.")
else:
    st.error("Customer not found in the dataset.")

