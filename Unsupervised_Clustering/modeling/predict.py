# predict.py
import logging
import joblib
import pandas as pd


def load_model(model_path="models/kmeans_model.pkl"):
    """Load a trained KMeans model from disk."""
    try:
        model = joblib.load(model_path)
        logging.info(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        raise


def predict_cluster(model, data):
    """Assign cluster labels to new input data."""
    try:
        labels = model.predict(data)
        logging.info("✅ Cluster prediction completed.")
        return labels
    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        raise
