import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import joblib
import os

# ----------------------------
# STEP 1: Load Dataset
# ----------------------------

data_path = os.path.join("data", "realistic_b2b_10000_dataset.xlsx")
df = pd.read_excel(data_path)

print("Dataset loaded successfully!")
print(df.head())

# ----------------------------
# STEP 2: Select Feature Columns
# ----------------------------

feature_cols = [
    "industry",
    "company_size",
    "website_visits",
    "pages_viewed",
    "time_spent_min",
    "email_opens",
    "content_downloads",
    "demo_request",
]

X = df[feature_cols]

# Create synthetic target
df["purchase"] = (
    (df["website_visits"] * 0.3)
    + (df["pages_viewed"] * 0.2)
    + (df["time_spent_min"] * 0.2)
    + (df["email_opens"] * 0.2)
    + (df["demo_request"] * 2)
)

df["purchase"] = (df["purchase"] > df["purchase"].median()).astype(int)

y = df["purchase"]

# ----------------------------
# STEP 3: One-Hot Encoding
# ----------------------------

X_encoded = pd.get_dummies(X, columns=["industry", "company_size"], drop_first=True)
feature_columns = X_encoded.columns.tolist()

# ----------------------------
# STEP 4: Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

# ----------------------------
# STEP 5: Train ML Model
# ----------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Purchase Intent Model Accuracy: {acc:.2f}")

# ----------------------------
# STEP 6: Train Clustering Model
# ----------------------------

cluster_features = [
    "website_visits",
    "pages_viewed",
    "time_spent_min",
    "email_opens",
    "content_downloads",
]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(df[cluster_features])

# ----------------------------
# STEP 7: Save Models
# ----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, os.path.join("models", "purchase_intent_model.pkl"))
joblib.dump(kmeans, os.path.join("models", "buyer_clusters.pkl"))
joblib.dump(feature_columns, os.path.join("models", "feature_columns.pkl"))
joblib.dump(cluster_features, os.path.join("models", "cluster_features.pkl"))

print("All models saved successfully!")
