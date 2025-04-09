# app.py

import streamlit as st
import pandas as pd
import logging
from Unsupervised_Clustering import config
from Unsupervised_Clustering.dataset import load_data
from Unsupervised_Clustering.features import select_features, scale_features
from Unsupervised_Clustering.modeling.train import (
    train_kmeans,
    calculate_wcss,
    calculate_silhouette,
)
from Unsupervised_Clustering.modeling.predict import load_model, predict_cluster
from Unsupervised_Clustering.plots import (
    plot_elbow,
    plot_silhouette,
    plot_cluster_scatter,
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.info("📊 Streamlit app loaded")

# Title
st.title("🧠 Mall Customer Segmentation using Unsupervised Learning")

# Sidebar
st.sidebar.header("Configuration")
cluster_count = st.sidebar.slider("Select number of clusters", 2, 10, 5)
logger.info(f"Cluster count selected: {cluster_count}")

# Load data
st.subheader("🔍 Raw Data")
df = load_data()
st.dataframe(df.head())
logger.info("Raw data loaded and displayed")

# Feature Selection
try:
    features = select_features(df)
    scaled_features = scale_features(features)
except Exception as e:
    st.error("❌ Feature selection or scaling failed.")
    logger.error(f"Feature processing error: {e}")


# Train model
st.subheader("📈 Training the Model")
try:
    model, labels, centers = train_kmeans(scaled_features, cluster_count)

    # Debugging Step 2: Add cluster labels and confirm
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

# Elbow and Silhouette Plots
st.subheader("📊 Evaluation Metrics")
wss_df = calculate_wcss(scaled_features)
sil_df = calculate_silhouette(scaled_features)
plot_elbow(wss_df)
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
