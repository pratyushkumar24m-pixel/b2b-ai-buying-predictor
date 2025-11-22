import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="B2B Intent Predictor",
    layout="wide"
)

# -------------------------
# PREMIUM SIMPLE NAVBAR
# -------------------------
def navbar():
    st.markdown("""
        <style>
            .navbar {
                background-color:#0A66C2;
                padding:14px;
                border-radius:8px;
                margin-bottom:15px;
            }
            .navbar a {
                padding: 10px 22px;
                color: white;
                font-size: 16px;
                text-decoration: none;
                margin-right: 10px;
            }
            .navbar a:hover {
                background-color:#004182;
                border-radius:5px;
            }
        </style>

        <div class="navbar">
            <a href="?page=home">Home</a>
            <a href="?page=upload">Upload & Predict</a>
            <a href="?page=about">About</a>
        </div>
    """, unsafe_allow_html=True)

navbar()

# Get page
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["home"])[0]


# -------------------------
# LOAD MODELS
# -------------------------
MODEL_DIR = "models"

purchase_model = joblib.load(os.path.join(MODEL_DIR, "purchase_intent_model.pkl"))
kmeans_model = joblib.load(os.path.join(MODEL_DIR, "buyer_clusters.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
cluster_features = joblib.load(os.path.join(MODEL_DIR, "cluster_features.pkl"))


# -------------------------
# HOME PAGE
# -------------------------
if page == "home":

    st.title("🧠 AI-Powered B2B Buying Intent Predictor")
    st.markdown("""
    A simple and intelligent tool to analyze B2B engagement data and predict:
    - Buying Intent (0 = Low, 1 = High)  
    - Behavioral Cluster  
    - Lead Score  
    - Buying Stage (Awareness → Decision)  
    
    Use the navigation above to upload your data.
    """)

    st.subheader("📥 Download Sample File")

    sample_file = "sample_b2b_dataset.csv"
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            st.download_button(
                label="Download sample_b2b_dataset.csv",
                data=f,
                file_name="sample_b2b_dataset.csv",
                mime="text/csv"
            )
    else:
        st.warning("sample_b2b_dataset.csv is missing. Add it to your folder.")



# -------------------------
# UPLOAD & PREDICT PAGE
# -------------------------
elif page == "upload":

    st.title("📊 Upload Data & Generate Predictions")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:

        # Load data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview of Data")
        st.dataframe(df.head())

        required_cols = [
            "industry", "company_size", "website_visits", "pages_viewed",
            "time_spent_min", "email_opens", "content_downloads", "demo_request"
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        # Encoding
        X = pd.get_dummies(df[required_cols],
                           columns=["industry", "company_size"],
                           drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Predictions
        df["predicted_intent"] = purchase_model.predict(X)
        df["cluster"] = kmeans_model.predict(df[cluster_features])

        # Lead score formula
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

        # Buying stage classification
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

        st.write("### Final Prediction Results")
        st.dataframe(df)

        st.download_button(
            label="Download Results CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="b2b_predictions_output.csv",
            mime="text/csv"
        )


# -------------------------
# ABOUT PAGE
# -------------------------
elif page == "about":

    st.title("ℹ️ About This Project")
    st.markdown("""
    This project predicts organizational buying intent using digital engagement signals.

    Developed by **Pratyush Kumar**  
    MBA – AI-driven B2B Market Analytics Project  
    """)


# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:#555; font-size:14px;'>
    Built with ❤️ and AI — Pratyush Kumar
</div>
""", unsafe_allow_html=True)
