import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ----------------------------
# Load Models
# ----------------------------

MODEL_DIR = "models"

purchase_model = joblib.load(os.path.join(MODEL_DIR, "purchase_intent_model.pkl"))
kmeans_model = joblib.load(os.path.join(MODEL_DIR, "buyer_clusters.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
cluster_features = joblib.load(os.path.join(MODEL_DIR, "cluster_features.pkl"))

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🧠 AI-Driven B2B Lead Score & Purchase Intent Predictor")

# Stylish Header Signature
st.markdown("<h4 style='color:#888; margin-top:-10px;'>By <b>Pratyush Kumar</b></h4>", 
            unsafe_allow_html=True)

st.subheader("Upload your dataset below to generate lead scores")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # ----------------------------
    # Preprocessing for Model
    # ----------------------------

    required_cols = [
        "industry", "company_size", "website_visits", "pages_viewed",
        "time_spent_min", "email_opens", "content_downloads", "demo_request"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
        st.stop()

    X = df[required_cols]

    # One-hot encode
    X_encoded = pd.get_dummies(X, columns=["industry", "company_size"], drop_first=True)

    # Align encoded columns with model's expected columns
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    # ----------------------------
    # Generate Predictions
    # ----------------------------

    df["predicted_intent"] = purchase_model.predict(X_encoded)

    # Clustering
    df["cluster"] = kmeans_model.predict(df[cluster_features])

    # ----------------------------
    # Lead Score (0–100)
    # ----------------------------

    raw_score = (
        df["website_visits"] * 0.8 +
        df["pages_viewed"] * 1.2 +
        df["time_spent_min"] * 0.5 +
        df["email_opens"] * 3 +
        df["content_downloads"] * 5 +
        df["demo_request"] * 20 +
        df["predicted_intent"] * 25
    )

    df["lead_score"] = ((raw_score - raw_score.min()) /
                        (raw_score.max() - raw_score.min()) * 100).round(0)

    # ----------------------------
    # Display Results
    # ----------------------------

    st.write("### Final Output with Lead Score (0–100)")
    st.dataframe(df[["industry", "company_size", "predicted_intent", "cluster", "lead_score"]].head())

    # ----------------------------
    # Download Output
    # ----------------------------

    st.write("### Download Full Results")

    st.download_button(
        label="📥 Download Predictions CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_output_with_lead_score.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV or Excel file to begin.")


# ---------------------------------------------------
# Stylish Signature Footer
# ---------------------------------------------------

st.markdown("""
<hr style='border: 1px solid #666;'>

<div style='text-align:center; font-size:18px; color:#444; margin-top:10px;'>
   Engineered with ❤️ and AI — <b>Pratyush Kumar</b>
</div>
""", unsafe_allow_html=True)



