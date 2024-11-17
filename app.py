import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.exceptions import NotFittedError

# Load pre-trained models
models = {
    'Decision Tree': ('best_decision_tree_model.pkl', 'decision_tree_model_results.npy'),
    'Logistic Regression': ('best_logistic_pipeline.pkl', 'logistic_pipeline_results.npy'),
    'Logistic Regression (L2)': ('best_lr_model.pkl', 'lr_model_results.npy'),
    'Calibrated SVC': ('best_svc_pipeline.pkl', 'svc_pipeline_results.npy')
}

# Load preprocessing artifacts
preprocessor = joblib.load('preprocessor_pipeline.pkl')
original_columns = joblib.load('original_columns.pkl')
metadata = joblib.load('preprocessing_metadata.pkl')

# Streamlit title and file uploader
st.title("Customer Purchase Prediction")
uploaded_file = st.file_uploader("Upload your dataset (.xlsx)", type="xlsx")

if uploaded_file:
    # Load data
    data = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data.head())
    
    # Drop `Purchase Probability` if it exists
    if 'Purchase Probability' in data.columns:
        data = data.drop(columns=['Purchase Probability'])
        st.write("`Purchase Probability` dropped from the dataset for prediction.")

    # Check column alignment
    missing_cols = set(original_columns) - set(data.columns)
    if missing_cols:
        st.error(f"The uploaded dataset is missing the following columns: {missing_cols}")
    else:
        # Align column order
        data = data[original_columns]

        # Preprocess data
        try:
            preprocessed_data = preprocessor.transform(data)
            st.success("Data successfully preprocessed!")
        except NotFittedError as e:
            st.error(f"Error in preprocessing: {e}")
            st.stop()

        # Model selection
        model_name = st.selectbox("Select a model for prediction:", list(models.keys()))
        if model_name:
            # Load the selected model and its threshold
            model_file, results_file = models[model_name]
            model = joblib.load(model_file)
            results = np.load(results_file, allow_pickle=True).item()
            best_threshold = results.get('best_threshold', 0.5)

            # Prediction
            if st.button("Predict"):
                proba = model.predict_proba(preprocessed_data)[:, 1]
                predictions = (proba >= best_threshold).astype(int)

                # Display results
                data['Prediction'] = ['Yes' if pred else 'No' for pred in predictions]
                data['Likelihood (%)'] = (proba * 100).round(2)
                st.write("Prediction Results:")
                st.write(data[['CUSTOMERNAME', 'Prediction', 'Likelihood (%)']])

                # Download results
                output_file = 'prediction_results.xlsx'
                data.to_excel(output_file, index=False)
                with open(output_file, 'rb') as f:
                    st.download_button(
                        label="Download Prediction Results",
                        data=f,
                        file_name=output_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
