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

        # Create an interactive bar chart using Plotly
        fig = px.bar(
            yearly_transactions,
            x="YEAR",
            y="Total Amount Purchased",
            color="YEAR",
            title=f"Yearly Transactions for {customer_name}",
            text_auto=True,
            template="plotly_dark",
        )

        # Update layout for a professional look
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Total Amount Purchased (AED)",
            coloraxis_showscale=False,
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display additional customer insights
        st.markdown("---")
        st.subheader("📋 Additional Customer Insights")

        # Prepare the data for the table
        insights_data = {
            "Metric": [
                "Customer Lifetime",
                "Months Since Last Purchase",
                "Max Time Without Purchase",
                "Trend Classification",
                "Average Purchase Value (AED)",
                "Total Transactions",
            ],
            "Value": [
                customer_transactions["Customer_Lifetime"].iloc[0],
                customer_transactions["Months_Since_Last_Purchase"].iloc[0],
                customer_transactions["Max_Time_Without_Purchase"].iloc[0],
                customer_transactions["Trend_Classification"].iloc[0],
                f"{round(customer_transactions['Average_Purchase_Value'].iloc[0], 2):,.2f} AED",
                customer_transactions["Customer_Transactions"].iloc[0],
            ],
        }

        # Create a Plotly Table with conditional formatting
        table_fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["<b>Metric</b>", "<b>Value</b>"],
                        fill_color="#1F2937",
                        font=dict(color="white", size=14),
                        align="center",
                    ),
                    cells=dict(
                        values=[
                            insights_data["Metric"],
                            insights_data["Value"],
                        ],
                        fill_color="#4B5563",
                        font=dict(color="white", size=12),
                        align="center",
                    ),
                )
            ]
        )

        table_fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))

        # Display the table
        st.plotly_chart(table_fig, use_container_width=True)

    else:
        st.warning(f"No transaction data available for {customer_name}.")
else:
    st.error("Customer not found in the dataset.")


