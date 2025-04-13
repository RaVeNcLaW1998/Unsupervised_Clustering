# app.py

import streamlit as st
import logging
from Unsupervised_Clustering.dataset import load_data
from Unsupervised_Clustering.features import select_features, scale_features
from Unsupervised_Clustering.modeling.train import (
    train_kmeans,
    calculate_silhouette,
)
from Unsupervised_Clustering.modeling.predict import load_model, predict_cluster
from Unsupervised_Clustering.plots import (
    plot_silhouette,
    plot_cluster_scatter,
)

from sklearn.preprocessing import StandardScaler

# Initialize logger
logger = logging.getLogger(__name__)
logger.info("📊 Streamlit app loaded")

st.title("🧠 Mall Customer Segmentation using Unsupervised Learning")

# 🔍 Predict a new customer's cluster
st.subheader("🎯 Predict Customer Segment")

with st.form("predict_form"):
    age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
    income = st.number_input(
        "Enter Annual Income", min_value=0, max_value=200, value=50
    )
    score = st.number_input(
        "Enter Spending Score", min_value=0, max_value=100, value=50
    )
    submitted = st.form_submit_button("Predict Segment")


# Feature Selection
try:
    df = load_data()
    features = select_features(df)
    scaled_features = scale_features(features)
except Exception as e:
    st.error("❌ Feature selection or scaling failed.")
    logger.error(f"Feature processing error: {e}")

# Make prediction if form was submitted
if submitted:
    try:
        model = load_model("models/kmeans_model.pkl")
        scaler = StandardScaler()
        scaler.fit(features)
        input_scaled = scaler.transform([[income, score]])
        cluster = predict_cluster(model, input_scaled)[0]
        st.success(f"🎉 This customer belongs to Cluster: {cluster}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.error(f"Prediction failed: {e}")


# Sidebar
st.sidebar.header("Configuration")
cluster_count = st.sidebar.slider("Select number of clusters", 2, 10, 5)
logger.info(f"Cluster count selected: {cluster_count}")

# Train model
st.subheader("📈 Training the Model")
try:
    model, labels, centers = train_kmeans(scaled_features, cluster_count)
    df["Cluster"] = labels
    st.success("✅ Model trained and cluster labels assigned!")
    logger.info("KMeans model trained and clusters assigned")
except Exception as e:
    logger.error(f"Model training failed: {e}")
    st.error("❌ Model training failed. Check logs for details.")

if "Cluster" not in df.columns:
    st.warning("⚠️ 'Cluster' column not found in DataFrame!")


# Visualize cluster scatter
st.subheader("🌀 Customer Clusters")
try:
    plot_cluster_scatter(df)
except Exception as e:
    logger.error(f"❌ Error in plot_cluster_scatter: {e}")
    st.error("Plotting error: Could not render cluster scatter plot.")

# Silhouette Plot only
st.subheader("📊 Silhouette Score")
sil_df = calculate_silhouette(scaled_features)
plot_silhouette(sil_df)

# Download segmented data
st.subheader("⬇️ Download Segmented Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="segmented_customers.csv",
    mime="text/csv",
)
logger.info("Segmented customer data prepared for download")

# Footer
st.markdown("---")
