import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pre-trained models and preprocessing artifacts
models = {
    'Decision Tree': ('best_decision_tree_model.pkl', 'decision_tree_model_results.npy'),
    'Logistic Regression': ('best_logistic_pipeline.pkl', 'logistic_pipeline_results.npy'),
    'Logistic Regression (L2)': ('best_lr_model.pkl', 'lr_model_results.npy'),
    'Calibrated SVC': ('best_svc_pipeline.pkl', 'svc_pipeline_results.npy')
}

# Load preprocessing pipeline and metadata
preprocessor = joblib.load('preprocessor_pipeline.pkl')
original_columns = joblib.load('original_columns.pkl')

# Load customer data for prediction
data = pd.read_excel('customer_transformed_data_with_cltv.xlsx')

# Remove the 'Purchase Probability' column for prediction
if 'Purchase Probability' in data.columns:
    data = data.drop(columns=['Purchase Probability'])

# Align columns to the original structure
data = data[original_columns]

# Streamlit app setup
st.title("Customer Purchase Prediction")

# Display the input data
st.write("Loaded Customer Data for Prediction:")
st.write(data.head())

# Preprocess the data
st.write("Preprocessing the data...")
preprocessed_data = preprocessor.transform(data)
st.success("Data successfully preprocessed!")

# Model selection
model_name = st.selectbox("Select a model for prediction:", list(models.keys()))
if model_name:
    # Load the selected model and its best threshold
    model_file, results_file = models[model_name]
    model = joblib.load(model_file)
    results = np.load(results_file, allow_pickle=True).item()
    best_threshold = results.get('best_threshold', 0.5)

    # Predict
    if st.button("Predict"):
        proba = model.predict_proba(preprocessed_data)[:, 1]
        predictions = (proba >= best_threshold).astype(int)

        # Append predictions to the original data
        data['Prediction'] = ['Yes' if pred else 'No' for pred in predictions]
        data['Likelihood (%)'] = (proba * 100).round(2)

        # Display results
        st.write("Prediction Results:")
        st.write(data[['CUSTOMERNAME', 'Prediction', 'Likelihood (%)']])

        # Allow downloading the results
        output_file = 'prediction_results.xlsx'
        data.to_excel(output_file, index=False)
        with open(output_file, 'rb') as f:
            st.download_button(
                label="Download Prediction Results",
                data=f,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

