import streamlit as st
from utils.theme import load_css, section_header, metric_card, show_image

st.set_page_config(page_title="Demand Forecasting | Retail Ops", layout="wide")
load_css()

section_header(
    "Demand Forecasting",
    "Forecasting daily purchase trends to anticipate demand and plan operations. Input: daily demand (Oct–Nov). Forecast period: December."
)

col1, col2 = st.columns(2)
with col1:
    metric_card("Mean absolute error", "14,470.99", "blue")
with col2:
    metric_card("RMSE", "29,962.52", "amber")

st.markdown("""
These metrics indicate the model's average and squared error over the test period.
Prophet effectively captures seasonality and trends in retail demand data.
""")

st.markdown("### Platform daily purchase forecast: Oct–Nov actuals + Dec forecast")
show_image("phase3b_forecast_plot_with_dec_line_full.png", caption="Full forecast: actuals + December projection")

col1, col2 = st.columns(2)
with col1:
    show_image("Demand Trend October–November 2019.png", caption="Demand trend, Oct–Nov 2019")
with col2:
    show_image("Cumulative Predicted Purchases – December.png", caption="Cumulative predicted purchases, December")