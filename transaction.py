import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
st.title("Customer Purchase Prediction Viewer 🎯")

# Model Selection
selected_model = st.selectbox("Select a Model", list(model_columns.keys()))

# Sidebar Metrics
st.sidebar.header("📊 Model Metrics")
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

    # Display outcomes with colorful badges
    prediction_text = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.markdown(f"### Prediction: **:green[{prediction_text}]**" if prediction == 1 else f"### Prediction: **:red[{prediction_text}]**")
    st.markdown(f"### Probability: **:blue[{probability:.2f}%]**")

    # Display transaction-level data for the selected customer
    customer_transactions = transaction_data[transaction_data["CUSTOMERNAME"] == customer_name]

    if not customer_transactions.empty:
        st.subheader("📈 Transaction-Level Insights")

        # Group transactions by YEAR
        yearly_transactions = (
            customer_transactions.groupby("YEAR")["Total Amount Purchased"]
            .sum()
            .reset_index()
        )

        # Create an interactive line plot using Plotly
        fig = px.line(
            yearly_transactions,
            x="YEAR",
            y="Total Amount Purchased",
            markers=True,
            title=f"Yearly Transactions for {customer_name}",
            template="plotly_white",
            text="Total Amount Purchased",
        )

        # Update layout for a professional look
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Total Amount Purchased (AED)",
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display additional customer insights
        st.markdown("---")
        st.subheader("📋 Additional Customer Insights")

        # Display customer insights as separate metrics or bar charts
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Customer Lifetime (Months)", customer_transactions["Customer_Lifetime"].iloc[0])
            st.metric("Months Since Last Purchase", customer_transactions["Months_Since_Last_Purchase"].iloc[0])

        with col2:
            st.metric("Max Time Without Purchase (Months)", customer_transactions["Max_Time_Without_Purchase"].iloc[0])
            st.metric("Total Transactions", customer_transactions["Customer_Transactions"].iloc[0])

        # Display a bar chart for trend classification and average purchase value
        st.markdown("### Trend Classification & Average Purchase Value")

        bar_chart_data = pd.DataFrame({
            "Metric": ["Trend Classification", "Average Purchase Value (AED)"],
            "Value": [
                customer_transactions["Trend_Classification"].iloc[0],
                customer_transactions["Average_Purchase_Value"].iloc[0],
            ],
        })

        trend_chart = px.bar(
            bar_chart_data,
            x="Metric",
            y="Value",
            title="Customer Insights",
            text="Value",
            template="plotly_white",
        )

        st.plotly_chart(trend_chart, use_container_width=True)

    else:
        st.warning(f"No transaction data available for {customer_name}.")
else:
    st.error("Customer not found in the dataset.")

