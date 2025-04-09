# features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from Unsupervised_Clustering import config


def select_features(df):
    """Select numerical features for clustering (e.g., Annual Income and Spending Score)."""
    try:
        features = df.iloc[
            :, 2:4
        ]  # Assumes 'Age', 'Annual Income', 'Spending Score' start at col 2
        logging.info("Selected features for clustering.")
        return features
    except Exception as e:
        logging.error(f"Error selecting features: {e}")
        raise


def scale_features(features):
    """Standardize the features before clustering."""
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        logging.info("Features scaled successfully.")
        return scaled
    except Exception as e:
        logging.error(f"Error scaling features: {e}")
        raise
