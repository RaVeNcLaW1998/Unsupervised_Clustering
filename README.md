# 🧠 Mall Customer Segmentation using Unsupervised Learning

This Streamlit project performs customer segmentation for a shopping mall using **KMeans clustering**. It allows interactive exploration of customer data and visualization of clustering results.

---

## 📂 Project Structure

```
Unsupervised_Clustering/
│
├── data/                   # Contains raw, processed datasets
│   └── raw/
│       └── mall_customers.csv
│
├── models/                 # Trained model artifacts (.pkl)
│
├── Unsupervised_Clustering/
│   ├── dataset.py          # Loads and preprocesses data
│   ├── features.py         # Feature selection and scaling
│   ├── modeling/
│   │   ├── train.py        # KMeans training and evaluation
│   │   └── predict.py      # Load model and predict clusters
│   ├── plots.py            # Visualizations
│   └── config.py           # Logging config
│
├── app.py                  # Streamlit front-end
├── requirements.txt        # Dependencies
└── README.md               # You're reading it now
```

---

## 🚀 How to Run the App

1. ✅ Install required dependencies:

```bash
pip install -r requirements.txt
```

2. ✅ Run the app:

```bash
streamlit run app.py
```

3. 📊 Open the app in your browser at `http://localhost:8501`

---

## ⚙️ Features

- 📥 Upload & load raw data (`mall_customers.csv`)
- 🔍 Feature selection & scaling
- 📈 KMeans clustering with interactive cluster count selection
- 🎨 Visualizations:
  - Cluster scatter plot
  - Elbow plot (WSS)
  - Silhouette score plot
- 📤 Download segmented customer data with cluster labels

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
```

---

## 🧠 Authors

Built with ❤️ by Athul Krishna Radhakrishnan Nair

## Streamlit App

https://unsupervisedclustering-x3juzpyzmodddkkaa3hu6w.streamlit.app/
