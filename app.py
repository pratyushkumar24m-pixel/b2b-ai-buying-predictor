import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI B2B Intent Predictor",
    layout="wide"
)

# ---------------------------------------------------------------------
# CUSTOM CSS FOR PREMIUM WEBSITE LOOK
# ---------------------------------------------------------------------
st.markdown("""
    <style>
        body {
            background-color: #F8FAF8;
        }
        .hero {
            background-color: #E6F4EA;
            border-radius: 18px;
            padding: 50px;
            margin-top: -30px;
            margin-bottom: 40px;
        }
        .hero-title {
            font-size: 40px;
            font-weight: 650;
            color: #2A3F2F;
            line-height: 1.2;
        }
        .hero-sub {
            font-size: 18px;
            color: #4C6B57;
            margin-top: 10px;
            margin-bottom: 25px;
        }
        .about-section {
            background-color: #FFFFFF;
            padding: 40px;
            border-radius: 15px;
            margin-top: 20px;
        }
        .section-title {
            font-size: 30px;
            font-weight: 600;
            color: #2A3F2F;
            margin-bottom: 10px;
        }
        .section-text {
            font-size: 16px;
            color: #4B4B4B;
            line-height: 1.6;
        }
        .navbar {
            background-color: #0A66C2;
            padding: 14px;
            border-radius: 8px;
            margin-bottom: 20px;
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
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# NAVIGATION BAR
# ---------------------------------------------------------------------
def navbar():
    st.markdown("""
        <div class="navbar">
            <a href="?page=home">Home</a>
            <a href="?page=about">About</a>
        </div>
    """, unsafe_allow_html=True)

navbar()

# Read navigation page
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["home"])[0]

# ---------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------
MODEL_DIR = "models"

purchase_model = joblib.load(os.path.join(MODEL_DIR, "purchase_intent_model.pkl"))
kmeans_model = joblib.load(os.path.join(MODEL_DIR, "buyer_clusters.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
cluster_features = joblib.load(os.path.join(MODEL_DIR, "cluster_features.pkl"))


# =====================================================================
#  HOME PAGE (Hero + Upload)
# =====================================================================
if page == "home":

    # --- HERO SECTION ---
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
            <div class="hero">
                <div class="hero-title">Predict B2B Buying Intent <br> The Smart & Simple Way</div>
                <div class="hero-sub">
                    Upload your engagement dataset to instantly generate:<br>
                    • Lead Score • Buying Intent • Buyer Cluster • Buying Stage
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Upload inside hero
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("### Data Preview")
            st.dataframe(df.head())

            required_cols = [
                "industry", "company_size", "website_visits", "pages_viewed",
                "time_spent_min", "email_opens", "content_downloads", "demo_request"
            ]

            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()

            # Encode
            X = pd.get_dummies(df[required_cols], columns=["industry", "company_size"], drop_first=True)
            X = X.reindex(columns=feature_columns, fill_value=0)

            # Predictions
            df["predicted_intent"] = purchase_model.predict(X)
            df["cluster"] = kmeans_model.predict(df[cluster_features])

            # Lead Score
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

            # Buying stage
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

            st.write("### Prediction Results")
            st.dataframe(df)

            st.download_button(
                label="Download Predictions CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="b2b_predictions_output.csv",
                mime="text/csv"
            )

    # Illustration on right
    with col2:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/1048/1048949.png",
            use_column_width=True,
            caption="AI-powered B2B Insights"
        )

    # Sample dataset download
    st.write("### 📥 Download Sample Dataset")
    sample_file = "sample_b2b_dataset.csv"
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            st.download_button(
                label="Download sample_b2b_dataset.csv",
                data=f,
                file_name="sample_b2b_dataset.csv",
                mime="text/csv"
            )

# =====================================================================
#  ABOUT PAGE
# =====================================================================
elif page == "about":

    st.markdown("""
        <div class="about-section">
            <div class="section-title">About This Project</div>
            <div class="section-text">
                This AI-driven B2B Intent Prediction Tool was created to analyze digital 
                engagement patterns such as website activity, content downloads, email interactions, 
                and demo requests. Using machine learning models including Logistic Regression 
                and K-Means Clustering, the system predicts buying intent, behavioral clusters, 
                lead score, and the buyer’s stage in the decision journey.
                <br><br>
                Future enhancements may include real-time CRM integration, predictive revenue 
                modeling, and automated marketing recommendations for sales teams and B2B marketers.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:#555; font-size:14px;'>
    Built with ❤️ and AI — Pratyush Kumar
</div>
""", unsafe_allow_html=True)
