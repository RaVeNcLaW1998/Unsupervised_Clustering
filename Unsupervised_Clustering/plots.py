# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from Unsupervised_Clustering import config
import streamlit as st


def plot_pairplot(df):
    """Plot pairplot of numerical features."""
    try:
        sns.pairplot(df[["Age", "Annual_Income", "Spending_Score"]])
        plt.show()
        logging.info("Pairplot generated successfully.")
    except Exception as e:
        logging.error(f"Error generating pairplot: {e}")


def plot_cluster_scatter(df):
    """Plot scatter of Annual Income vs Spending Score with cluster hue."""
    try:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x="Annual_Income",
            y="Spending_Score",
            hue="Cluster",
            palette="colorblind",
            data=df,
            ax=ax,
        )
        ax.set_title("Customer Segmentation")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Failed to plot scatter: {e}")


def plot_elbow(wss_df):
    """Plot elbow curve from WSS data."""
    try:
        fig, ax = plt.subplots()
        ax.plot(wss_df["cluster"], wss_df["WSS_Score"], marker="o")
        ax.set_xlabel("No. of Clusters")
        ax.set_ylabel("WSS Score")
        ax.set_title("Elbow Plot")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Failed to plot elbow: {e}")


def plot_silhouette(sil_df):
    """Plot silhouette scores for different cluster counts."""
    try:
        fig, ax = plt.subplots()
        ax.plot(
            sil_df["cluster"], sil_df["Silhouette_Score"], marker="s", color="green"
        )
        ax.set_xlabel("No. of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Plot")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Failed to plot silhouette: {e}")
