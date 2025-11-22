import streamlit as st
import pandas as pd
import joblib
import os

MODEL_DIR = "models"
CLUSTER_MODEL_PATH = os.path.join(MODEL_DIR, "buyer_clusters.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
PURCHASE_MODEL_PATH = os.path.join(MODEL_DIR, "purchase_intent_model.pkl")

# Load models
cluster_model = joblib.load(CLUSTER_MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
purchase_model = joblib.load(PURCHASE_MODEL_PATH)

# Load sample file (you can place it inside your project folder)
SAMPLE_FILE_PATH = "sample_b2b_dataset.xlsx"

st.title("B2B Buying Intent Prediction Tool")
st.write("Upload your dataset to generate intent predictions, behavioural clusters, and lead scores.")

# -----------------------------
# DOWNLOAD SAMPLE FILE BUTTON
# -----------------------------
with open(SAMPLE_FILE_PATH, "rb") as sample_file:
    st.download_button(
        label="📥 Download Sample Excel Template",
        data=sample_file,
        file_name="sample_b2b_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.write("Use the above Excel template to prepare and upload your dataset.")

# -----------------------------
# FILE UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Read input file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df)

    # Preprocess input
    input_data = df.copy()

    # Align input data
    input_encoded = pd.get_dummies(input_data)
    missing_cols = set(feature_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
        
    input_encoded = input_encoded[feature_columns]

    # Predictions
    clusters = cluster_model.predict(input_encoded)
    intent_pred = purchase_model.predict(input_encoded)

    lead_score = (
        input_data["website_visits"] * 0.3 +
        input_data["pages_viewed"] * 0.2 +
        input_data["email_opens"] * 0.2 +
        input_data["content_downloads"] * 0.2 +
        input_data["demo_request"] * 0.1
    )

    # Combine output
    input_data["predicted_intent"] = intent_pred
    input_data["cluster"] = clusters
    input_data["lead_score"] = lead_score.round(2)

    st.write("### Final Prediction Output")
    st.dataframe(input_data)

    # Download results
    st.download_button(
        label="⬇️ Download Results as Excel",
        data=input_data.to_csv(index=False),
        file_name="b2b_predictions_output.csv",
        mime="text/csv"
    )
