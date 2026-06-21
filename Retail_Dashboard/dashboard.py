import streamlit as st
from utils.theme import load_css, section_header, metric_card

# Must be the first Streamlit command
st.set_page_config(
    page_title="Retail Ops | Demand & Sustainability",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


def overview_page():
    load_css()

    section_header(
        "Retail Demand & Sustainability",
        "An end-to-end retail intelligence platform — purchase prediction, demand forecasting, sustainability analysis, and customer segmentation."
    )

    st.markdown("""
    This dashboard presents a full-stack, machine learning–powered solution that addresses key challenges in modern retail operations.
    It is the result of an end-to-end data-driven pipeline encompassing purchase behaviour prediction, platform-level demand forecasting,
    sustainability-driven category analysis, and customer intelligence.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Classifier accuracy", "94.8%", "green")
    with col2:
        metric_card("Forecast MAE (Dec)", "14,471", "blue")
    with col3:
        metric_card("Customer segments", "7", "amber")
    with col4:
        metric_card("Users profiled", "5.32M", "blue")

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        with st.container(border=True):
            st.markdown('<p style="font-weight:600; font-size:15px; margin:0 0 12px;">Intelligence modules</p>', unsafe_allow_html=True)
            st.markdown('<span class="status-dot status-green"></span><b>Live Prediction</b> — real-time Random Forest classifier with SHAP explainability', unsafe_allow_html=True)
            st.markdown('<span class="status-dot status-blue"></span><b>Customer Intelligence</b> — K-Means segmentation across buyers and non-buyers', unsafe_allow_html=True)
            st.markdown('<span class="status-dot status-amber"></span><b>Demand Forecasting</b> — Prophet projections for December capacity planning', unsafe_allow_html=True)
            st.markdown('<span class="status-dot status-red"></span><b>Sustainability</b> — overstock and understock risk flags by category', unsafe_allow_html=True)

    with colB:
        with st.container(border=True):
            st.markdown('<p style="font-weight:600; font-size:15px; margin:0 0 12px;">Technical stack</p>', unsafe_allow_html=True)
            st.markdown('<b>Frontend</b> — Streamlit, custom CSS', unsafe_allow_html=True)
            st.markdown('<b>Data</b> — pandas, NumPy', unsafe_allow_html=True)
            st.markdown('<b>Modeling</b> — scikit-learn (Random Forest, K-Means), Prophet, SHAP', unsafe_allow_html=True)
            st.markdown('<b>Evaluation</b> — accuracy, F1, ROC/PR-AUC, MAE, RMSE, silhouette score', unsafe_allow_html=True)
    

# ============================================================
# NAVIGATION — explicit st.Page registration so sidebar labels
# are fully under our control (independent of filenames).
# ============================================================
overview = st.Page(overview_page, title="Overview", icon="🏠", default=True)
live_prediction = st.Page("pages/1_Live_Prediction.py", title="Live Prediction", icon="⚡")
model_metrics = st.Page("pages/2_Model_Metrics.py", title="Model Metrics", icon="📊")
customer_intelligence = st.Page("pages/3_Customer_Intelligence.py", title="Customer Intelligence", icon="👥")
demand_forecasting = st.Page("pages/4_Demand_Forecasting.py", title="Demand Forecasting", icon="📈")
sustainability = st.Page("pages/5_Sustainability.py", title="Sustainability", icon="🌿")

pg = st.navigation([
    overview,
    live_prediction,
    model_metrics,
    customer_intelligence,
    demand_forecasting,
    sustainability,
])
pg.run()