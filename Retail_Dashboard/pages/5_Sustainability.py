import streamlit as st
from utils.theme import load_css, section_header, show_image

st.set_page_config(page_title="Sustainability | Retail Ops", layout="wide")
load_css()

section_header(
    "Sustainability & Inventory Insights",
    "Category-level forecasting identifies under/overstock conditions ahead of time."
)

col1, col2 = st.columns(2)
with col1:
    show_image("Top_Bottom_Categories_BarChart.png", caption="Top and bottom performing categories")
with col2:
    show_image("Number of Categories.png", caption="Category count breakdown")

st.markdown("### Weekly demand stackplot")
show_image("stacked_area_plot_top5.png", caption="Weekly demand, top 5 categories")