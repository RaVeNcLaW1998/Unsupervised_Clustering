# dataset.py
import pandas as pd
import logging
from Unsupervised_Clustering import config


def load_data():
    try:
        df = pd.read_csv("data/raw/mall_customers.csv")
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise
