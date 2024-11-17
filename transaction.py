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
st.title("Customer Purchase Insights Dashboard")

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
    prediction = customer_data[prediction_col].iloc[0]
    probability = customer_data[probability_col].iloc[0] * 100  # Convert to percentage

    # Display outcomes with badges
    prediction_text = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.markdown(f"### Prediction: **:green[{prediction_text}]**" if prediction == 1 else f"### Prediction: **:red[{prediction_text}]**")
    st.markdown(f"### Probability: **:blue[{probability:.2f}%]**")

    # Transaction data for selected customer
    customer_transactions = transaction_data[transaction_data["CUSTOMERNAME"] == customer_name]

    if not customer_transactions.empty:
        st.subheader("📈 Customer Insights")

        # Calculate additional metrics
        mean_time_between_purchases = customer_transactions["Mean_Time_Between_Purchases"].mean()
        most_purchased_item = customer_transactions["ITEMGROUPDESCRIPTION"].mode()[0] if not customer_transactions["ITEMGROUPDESCRIPTION"].mode().empty else "N/A"
        country = customer_transactions["COUNTRYNAME"].iloc[0]

        # Display metrics in styled cards
        def style_card(header, value, background_color):
            return f"""
            <div style="background-color: {background_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-size: 18px;">
                <strong>{header}</strong><br>{value}
            </div>
            """

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(style_card("Country", country, "#4CAF50"), unsafe_allow_html=True)
            st.markdown(style_card("Customer Lifetime (Months)", customer_transactions["Customer_Lifetime"].iloc[0], "#2196F3"), unsafe_allow_html=True)
        with col2:
            st.markdown(style_card("Mean Time Between Purchases (Months)", f"{mean_time_between_purchases:.2f}", "#FF9800"), unsafe_allow_html=True)
            st.markdown(style_card("Total Transactions", customer_transactions["Customer_Transactions"].iloc[0], "#9C27B0"), unsafe_allow_html=True)
        with col3:
            st.markdown(style_card("Most Purchased Item", most_purchased_item, "#F44336"), unsafe_allow_html=True)
            st.markdown(style_card("Average Purchase Value (AED)", f"{customer_transactions['Average_Purchase_Value'].iloc[0]:,.2f}", "#3F51B5"), unsafe_allow_html=True)

        # Group transactions by YEAR for a line chart
        yearly_transactions = (
            customer_transactions.groupby("YEAR")["Total Amount Purchased"]
            .sum()
            .reset_index()
        )

        # Interactive line plot with gradient color
        st.subheader("📊 Yearly Transactions")
        fig = px.line(
            yearly_transactions,
            x="YEAR",
            y="Total Amount Purchased",
            markers=True,
            title=f"Yearly Transactions for {customer_name}",
            template="plotly_white",
            line_shape="linear",
        )
        fig.update_traces(
            line_color="purple",
            mode="lines+markers",
        )
        fig.update_layout(
            xaxis=dict(tickmode="linear", tickformat=".0f"),  # Clean X-axis
            xaxis_title="Year",
            yaxis_title="Total Amount Purchased (AED)",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"No transaction data available for {customer_name}.")
else:
    st.error("Customer not found in the dataset.")

