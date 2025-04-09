# train.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from Unsupervised_Clustering import config
import joblib
import os


def train_kmeans(data, n_clusters=5):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        joblib.dump(kmeans, "models/kmeans_model.pkl")
        logging.info("✅ KMeans model training complete and saved.")

        return kmeans, labels, centers
    except Exception as e:
        logging.error(f"❌ Error training KMeans model: {e}")
        raise


def calculate_wcss(data, k_range=range(3, 9)):
    """Calculate Within-Cluster Sum of Squares for range of clusters."""
    try:
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            scores.append({"cluster": k, "WSS_Score": kmeans.inertia_})
        logging.info("WCSS scores calculated successfully.")
        return pd.DataFrame(scores)
    except Exception as e:
        logging.error(f"Error calculating WCSS scores: {e}")
        raise


def calculate_silhouette(data, k_range=range(3, 9)):
    """Calculate Silhouette Score for range of clusters."""
    try:
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append({"cluster": k, "Silhouette_Score": score})
        logging.info("Silhouette scores calculated successfully.")
        return pd.DataFrame(scores)
    except Exception as e:
        logging.error(f"Error calculating silhouette scores: {e}")
        raise
