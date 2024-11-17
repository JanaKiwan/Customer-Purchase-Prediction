import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# Load Models and Data
# =========================
st.title("Customer Purchase Prediction")

# Load processed data
try:
    customer_data = pd.read_excel("customer_agg_transformed_processed.xlsx")
    st.write("Customer data loaded successfully.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Load models and pipelines
model_files = {
    "Logistic Regression (L1)": ("best_logistic_pipeline.pkl", "logistic_pipeline_results.npy"),
    "Logistic Regression (L2)": ("best_lr_model.pkl", "lr_model_results.npy"),
    "SVC (Calibrated)": ("best_svc_pipeline.pkl", "svc_pipeline_results.npy"),
    "Decision Tree": ("best_decision_tree_model.pkl", "decision_tree_model_results.npy")
}

loaded_models = {}
for model_name, (model_path, results_path) in model_files.items():
    try:
        model = joblib.load(model_path)
        metrics = np.load(results_path, allow_pickle=True).item()
        loaded_models[model_name] = (model, metrics)
        st.write(f"Loaded model: {model_name}")
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        st.stop()

# =========================
# Feature Input
# =========================
st.sidebar.header("Input Features")

# Extract feature columns
feature_columns = customer_data.columns.tolist()
excluded_columns = ["CUSTOMERNAME"]
feature_columns = [col for col in feature_columns if col not in excluded_columns]

# User inputs
input_features = {}
for feature in feature_columns:
    if customer_data[feature].dtype in [np.int64, np.float64]:
        input_features[feature] = st.sidebar.number_input(feature, value=0.0)
    else:
        input_features[feature] = st.sidebar.selectbox(
            feature, customer_data[feature].unique()
        )

# Convert inputs to DataFrame
features = pd.DataFrame([input_features])

# =========================
# Prediction Logic
# =========================
def make_prediction(model, features, threshold=0.5):
    """
    Generates the prediction and probability.
    """
    proba = model.predict_proba(features)[:, 1]
    prediction = proba >= threshold
    return prediction, proba

st.header("Model Predictions")
selected_model_name = st.selectbox("Select a model to use", list(loaded_models.keys()))

if st.button("Predict"):
    model, metrics = loaded_models[selected_model_name]
    threshold = metrics.get("best_threshold", 0.5)

    try:
        prediction, proba = make_prediction(model, features, threshold)
        prediction_result = "Yes" if prediction[0] else "No"
        st.write(f"**Prediction Outcome:** {prediction_result}")
        st.write(f"**Likelihood of Purchase:** {proba[0] * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# =========================
# Additional Information
# =========================
st.header("Additional Information")
st.write("This application predicts the likelihood of customer purchases using multiple models.")
st.write("You can view the details of each model and its performance metrics.")
