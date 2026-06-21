# 🛍 Retail Demand & Sustainability Dashboard

This project presents an end-to-end machine learning–powered dashboard designed to optimize demand forecasting, purchase behavior prediction, and sustainability analysis for a retail platform.

Built using:
- 🐍 Python (v3.12)
- 📊 Streamlit for the interactive dashboard
- 🔍 Scikit-learn for classification modeling
- ⏱ Prophet for time series forecasting
- 🧹 Pandas, Matplotlib, Seaborn for preprocessing & visualizations

---

## 🚀 Project Overview

### 🔮 Phase 3A: Purchase Behavior Prediction
- Trained a Random Forest Classifier to predict whether a user will make a purchase during a session.
- Input features included session activity metrics such as views, cart actions, time spent, and price ranges.
- Model achieved:
  - Precision (class 1 - Purchase): 0.57
  - Recall (class 1 - Purchase): 0.95
  - F1-score (class 1): 0.71
  - Overall Accuracy: 94.8%
- Interactive simulation module included in dashboard for real-time classification based on input session attributes.

### 📈 Phase 3B: Demand Forecasting
- Used Prophet time series model to forecast platform-level purchases for December 2019.
- Trained on Oct–Nov daily demand.
- Achieved:
  - MAE: 14,471
  - RMSE: 29,963
- Visualizations: line chart, cumulative predictions, stacked weekly demand.

### 🌱 Phase 4: Sustainability Analysis
- Forecasted demand at product category level and compared with simulated supply.
- Generated top/bottom category demand charts and under/over-stock alerts.
- Supported sustainable inventory planning with visual insights.

---

## 📊 Visualizations
- Feature importance, ROC & PR curves, confusion matrix
- Forecast trends, demand decomposition
- Top/Bottom category bar charts, stacked area plots
- Interactive prediction tool with probability scores

---

## 🧠 Live Dashboard Components
- Overview page with methodology and visuals
- Real-time session classifier (live input form)
- Static model evaluation for 3A and 3B
- Sustainability stackplots and alerts

---

# 🛒 Retail Demand & Sustainability Dashboard

🚀 **Live Dashboard**: [Click here to view](https://retail-demand-sustainability-project-4apdgv6hxez4n5vonpgnb3.streamlit.app/)

This dashboard analyzes retail product demand trends and supports sustainable inventory management using machine learning and visual insights.



