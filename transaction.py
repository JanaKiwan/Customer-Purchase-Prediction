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
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e2f;
        color: white;
    }
    .stTextInput, .stSelectbox, .stButton, .stDataFrame {
        color: #1e1e2f !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("💼 Customer Purchase Insights Dashboard")

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
    st.markdown(
        f"<h3 style='color:green;'>Prediction: {prediction_text}</h3>"
        if prediction == 1
        else f"<h3 style='color:red;'>Prediction: {prediction_text}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<h4 style='color:blue;'>Probability: {probability:.2f}%</h4>", unsafe_allow_html=True)

    # Transaction data for selected customer
    customer_transactions = transaction_data[transaction_data["CUSTOMERNAME"] == customer_name]

    if not customer_transactions.empty:
        st.subheader("📈 Customer Insights")

        # Country Map
        country = customer_transactions["COUNTRYNAME"].iloc[0]
        map_fig = px.scatter_geo(
            locations=[country],
            locationmode="country names",
            text=[country],
            size=[5],
            title=f"Customer Country: {country}",
        )
        map_fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type="equirectangular",
            ),
            template="plotly_dark",
        )
        st.plotly_chart(map_fig)

        # Additional Insights
        mean_time_between_purchases = customer_transactions["Mean_Time_Between_Purchases"].mean()
        total_transactions = customer_transactions["Customer_Transactions"].iloc[0]
        avg_purchase_value = customer_transactions["Average_Purchase_Value"].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Time Between Purchases", f"{mean_time_between_purchases:.2f}")
        col2.metric("Total Transactions", total_transactions)
        col3.metric("Avg Purchase Value (AED)", f"{avg_purchase_value:,.2f} AED")

        # Yearly Transactions Plot
        yearly_transactions = (
            customer_transactions.groupby("YEAR")["Total Amount Purchased"]
            .sum()
            .reset_index()
        )
        st.subheader("📊 Yearly Transactions")
        fig = px.line(
            yearly_transactions,
            x="YEAR",
            y="Total Amount Purchased",
            title=f"Yearly Transactions for {customer_name}",
            template="plotly_dark",
            line_shape="spline",
            markers=True,
        )
        fig.update_traces(line_color="cyan")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"No transaction data available for {customer_name}.")
else:
    st.error("Customer not found in the dataset.")

