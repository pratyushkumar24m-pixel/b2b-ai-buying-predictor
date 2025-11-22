import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# PAGE CONFIG (Corporate SaaS Theme)
# ============================================
st.set_page_config(
    page_title="B2B Intent Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODELS
# ============================================
MODEL_DIR = "models"

purchase_model = joblib.load(os.path.join(MODEL_DIR, "purchase_intent_model.pkl"))
kmeans_model = joblib.load(os.path.join(MODEL_DIR, "buyer_clusters.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
cluster_features = joblib.load(os.path.join(MODEL_DIR, "cluster_features.pkl"))

# ============================================
# SIDEBAR (Navigation)
# ============================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Upload & Predict", "About"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Created by Pratyush Kumar**")
st.sidebar.markdown("AI-Driven B2B Analytics Project")


# ============================================
# HOME PAGE
# ============================================
if page == "Home":

    st.title("🧠 AI-Driven B2B Buying Intent & Lead Score Predictor")
    st.markdown("""
    Welcome to the B2B Intent Prediction Dashboard.

    This tool uses machine learning and behavioral analytics to estimate:
    - Buying Intent (0 = Low, 1 = High)  
    - Engagement Cluster (0, 1, 2)  
    - Lead Score (0–100)  
    - B2B Buying Stage (Awareness → Decision)  

    Use the sidebar to upload your dataset and generate predictions.
    """)

    st.markdown("### 📥 Download Sample Excel Template")

    sample_file = "sample_b2b_dataset.xlsx"
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            st.download_button(
                label="Download Sample File",
                data=f,
                file_name="sample_b2b_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("sample_b2b_dataset.xlsx not found. Add it to your app folder.")


# ============================================
# UPLOAD & PREDICT PAGE
# ============================================

elif page == "Upload & Predict":

    st.title("📊 Upload Data & Generate Predictions")

    st.markdown("""
    Upload your CSV/Excel file containing B2B engagement data.
    The model will calculate:
    - Predicted Intent  
    - Engagement Cluster  
    - Lead Score  
    - Buying Stage  
    """)

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:

        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        required_cols = [
            "industry", "company_size", "website_visits", "pages_viewed",
            "time_spent_min", "email_opens", "content_downloads", "demo_request"
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # One-hot encode
        X = pd.get_dummies(df[required_cols], columns=["industry", "company_size"], drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Predictions
        df["predicted_intent"] = purchase_model.predict(X)
        df["cluster"] = kmeans_model.predict(df[cluster_features])

        # Lead score calc
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

        # Buying Stage Mapping
        def buying_stage(score):
            if score < 30:
                return "Awareness"
            elif score < 60:
                return "Interest"
            elif score < 85:
                return "Evaluation"
            else:
                return "Decision"

        df["buying_stage"] = df["lead_score"].apply(buying_stage)

        st.write("### Final Predictions")
        st.dataframe(df)

        st.download_button(
            label="📥 Download Results",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="b2b_predictions_output.csv",
            mime="text/csv"
        )


# ============================================
# ABOUT PAGE
# ============================================

elif page == "About":

    st.title("📘 About This Project")

    st.markdown("""
    This B2B analytics dashboard predicts organizational buying intent using:
    - Machine Learning  
    - Behavioral Engagement Analysis  
    - Clustering  
    - Lead Scoring  

    Developed as part of an MBA project by **Pratyush Kumar**.
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<hr>

<div style='text-align:center; color: #555;'>
    Engineered with ❤️ — Pratyush Kumar  
    <br>AI-Driven B2B Market Analytics Dashboard
</div>
""", unsafe_allow_html=True)
