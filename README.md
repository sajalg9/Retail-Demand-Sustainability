# Retail Demand & Sustainability Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Status](https://img.shields.io/badge/status-live-success)

An end-to-end retail intelligence platform built on ~24 million real e-commerce events — purchase prediction, demand forecasting, sustainability risk analysis, and customer segmentation, deployed as an interactive Streamlit dashboard.

🚀 **Live Demo:** [Open Dashboard](https://retail-dashboard-jtx9fh8ywanswe8z2nwjui.streamlit.app/)

---

## What it does

Raw session-level e-commerce events (October–November 2019) feed four connected analytical layers:

- **Purchase prediction** — a Random Forest classifier predicts whether a session ends in a purchase, with SHAP explaining exactly which features drove any individual prediction.
- **Demand forecasting** — a Prophet model forecasts December platform demand from Oct–Nov history.
- **Sustainability analysis** — per-category forecasts are checked against recent demand to flag overstock and understock risk.
- **Customer intelligence** — RFM-style behavioral features and K-Means clustering segment all 5.3M users into seven groups, each paired with a recommended action.

The dashboard has six pages — **Overview, Live Prediction, Model Metrics, Customer Intelligence, Demand Forecasting, Sustainability** — all sharing one design system so it reads as a single product rather than five separate exports.

---

## Results

| Component | Metric | Value |
|---|---|---|
| Purchase classifier | Accuracy | **94.8%** |
| | F1 (no purchase / purchase) | 0.97 / 0.71 |
| | ROC AUC / PR AUC | 0.99 / 0.92 |
| Demand forecast | MAE / RMSE | 14,470.99 / 29,962.52 |
| Segmentation | Users profiled | 5,316,561 |
| | Buyers | 690,803 (13%) |

Clustering was run separately for buyers and non-buyers rather than on the full population, since the ~87/13 non-buyer/buyer split would otherwise have dominated the result.

One real finding worth calling out: the two highest-value buyer segments share almost identical spend and conversion rates — the only meaningful difference is recency, splitting "currently active" customers from the same kind of customer gone quiet for ~43 days. That distinction drives the win-back recommendation surfaced in the dashboard.

---

## Stack

Python · scikit-learn (Random Forest, K-Means) · Prophet · SHAP · pandas · matplotlib/seaborn · Streamlit — deployed on Streamlit Community Cloud.

---

## Structure

```
├── Notebooks/                # Preprocessing, feature engineering, modeling, sustainability analysis
├── Retail_Dashboard/         # Deployed Streamlit app
│   ├── dashboard.py          # Entry point + navigation
│   ├── pages/                # One file per dashboard page
│   ├── utils/theme.py        # Shared UI components
│   ├── assets/style.css
│   └── *.pkl / *.csv         # Trained models and segment data
└── README.md
```

---

## Running locally

```bash
git clone https://github.com/sajalg9/Retail-Demand-Sustainability.git
cd Retail-Demand-Sustainability/Retail_Dashboard
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard.py
```

Built and tested on **Python 3.11** with `numpy<2` — SHAP's compiled dependencies aren't yet stable on newer Python releases.

---

## Data

[E-Commerce Behavior Data from a Multi-Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) (Kaggle), October–November 2019.
