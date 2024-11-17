import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Models and Pipelines
# =========================
# Load preprocessing pipeline, original column structure, and preprocessing metadata
preprocessor_pipeline = joblib.load("preprocessor_pipeline.pkl")
original_columns = joblib.load("original_columns.pkl")
preprocessing_metadata = joblib.load("preprocessing_metadata.pkl")

# Load models and their results
models = {
    "Logistic Regression (L1)": {
        "model": joblib.load("best_logistic_pipeline.pkl"),
        "metrics": np.load("logistic_pipeline_results.npy", allow_pickle=True).item()
    },
    "Logistic Regression (L2)": {
        "model": joblib.load("best_lr_model.pkl"),
        "metrics": np.load("lr_model_results.npy", allow_pickle=True).item()
    },
    "Support Vector Classifier (Calibrated)": {
        "model": joblib.load("best_svc_pipeline.pkl"),
        "metrics": np.load("svc_pipeline_results.npy", allow_pickle=True).item()
    },
    "Decision Tree": {
        "model": joblib.load("best_decision_tree_model.pkl"),
        "metrics": np.load("decision_tree_model_results.npy", allow_pickle=True).item()
    }
}

# =========================
# Load Dataset
# =========================
# Directly load the dataset for predictions
data = pd.read_excel("customer_transformed_data_with_cltv.xlsx")
st.title("Customer Purchase Prediction")
st.write("Loaded Dataset Preview:")
st.dataframe(data.head())

# Align columns to the original structure
data = data[original_columns["categorical_vars"] + original_columns["numerical_vars"]]

# =========================
# Model Selection
# =========================
st.sidebar.title("Model Selection")
selected_model_name = st.sidebar.selectbox("Select a Model for Prediction:", list(models.keys()))
selected_model = models[selected_model_name]["model"]
selected_metrics = models[selected_model_name]["metrics"]
st.sidebar.write(f"Selected Model: **{selected_model_name}**")

# =========================
# Prediction
# =========================
if st.button("Predict"):
    # Preprocess the data
    preprocessed_data = preprocessor_pipeline.transform(data)

    # Generate predictions and probabilities
    predictions = selected_model.predict(preprocessed_data)
    probabilities = selected_model.predict_proba(preprocessed_data)[:, 1]

    # Apply the best threshold from the model's training
    threshold = selected_metrics.get("best_threshold", 0.5)
    prediction_outcome = (probabilities >= threshold).astype(int)

    # Append results to the dataset
    data["Purchase Prediction"] = prediction_outcome
    data["Purchase Probability"] = probabilities

    # Display the predictions
    st.write("Prediction Results:")
    st.dataframe(data)

    # Download link for results
    st.download_button(
        label="Download Predictions",
        data=data.to_csv(index=False),
        file_name="customer_predictions.csv",
        mime="text/csv"
    )

