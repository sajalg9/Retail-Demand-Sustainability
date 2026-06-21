import streamlit as st
import pandas as pd
from utils.theme import load_css, section_header, metric_card, show_image

st.set_page_config(page_title="Model Metrics | Retail Ops", layout="wide")
load_css()

section_header(
    "Model Performance",
    "Random Forest classifier, evaluated on held-out session data.",
    icon="model"
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    metric_card("Accuracy", "94.8%", "green", bar_pct=94.8)
with col2:
    metric_card("F1 no purchase", "0.97", "blue", bar_pct=97)
with col3:
    metric_card("F1 purchase", "0.71", "amber",  bar_pct=71)
with col4:
    metric_card("ROC AUC", "0.99", "green",  bar_pct=99)
with col5:
    metric_card("PR AUC", "0.92", "blue", bar_pct=92)

st.markdown("#### Evaluation metrics by class")
metrics_df = pd.DataFrame({
    "Class": ["0 (No purchase)", "1 (Purchase)"],
    "Precision": ["1.00", "0.57"],
    "Recall": ["0.95", "0.95"],
    "F1-score": ["0.97", "0.71"]
})
metrics_df.index = metrics_df.index + 1
st.dataframe(metrics_df, use_container_width=True)

st.markdown("""
The model is highly precise and reliable at detecting non-purchase behavior, and balances good recall for predicting actual purchases.
The high F1-score for class 0 reflects excellent precision and recall, while for class 1, high recall supports targeting purchase intent.
""")

col1, col2 = st.columns(2)
with col1:
    show_image("3A_Feature Importance.png", caption="Feature importance")
with col2:
    show_image("confusion_matrix.png", caption="Confusion matrix")

col3, col4 = st.columns(2)
with col3:
    show_image("ROC Curve.png", caption="ROC curve")
with col4:
    show_image("Precision-Recall Curve.png", caption="Precision-recall curve")

st.markdown("#### Session distribution")
show_image("hourly_activity_distribution.png", caption="Hourly activity distribution")