import streamlit as st
import pandas as pd
from utils.theme import load_css, section_header, metric_card, status_card

st.set_page_config(page_title="Customer Intelligence | Retail Ops", layout="wide")
load_css()

section_header(
    "Customer Intelligence",
    "All users grouped by recency, purchase rate, spend, and engagement — built with RFM-style features and K-Means clustering, run separately for buyers and non-buyers."
)

@st.cache_data
def load_segment_summary():
    return pd.read_csv("phase5_segment_profile_summary.csv")

profile_df = load_segment_summary()

total_users = profile_df["user_count"].sum()
buyer_pct = profile_df[profile_df["segment_group"] == "buyer"]["pct_of_total_users"].sum()

col1, col2, col3 = st.columns(3)
with col1:
    metric_card("Total users", f"{total_users:,.0f}", "blue")
with col2:
    metric_card("Buyers", f"{buyer_pct:.1f}%", "green")
with col3:
    metric_card("Segments", f"{len(profile_df)}", "amber")

st.markdown("### User distribution across segments")
chart_data = profile_df[["segment_name", "user_count"]].set_index("segment_name")
st.bar_chart(chart_data, use_container_width=True)

status_map = {
    "High-Value Active": "good",
    "High-Value Lapsed": "alert",
    "Frequent Browsers, Low Spend": "watch",
    "Occasional Low-Value": "watch",
    "Drive-by Visitors": "watch",
    "Engaged Browsers (No Cart)": "watch",
    "Cart Abandoners": "info",
}

section_header("Segment profiles & recommended actions")

buyer_segments = profile_df[profile_df["segment_group"] == "buyer"]
nonbuyer_segments = profile_df[profile_df["segment_group"] == "non_buyer"]

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Buyer segments")
    for _, row in buyer_segments.iterrows():
        status = status_map.get(row["segment_name"], "info")
        stats = [
            ("recency", f"{row['avg_recency_days']:.0f}d"),
            ("order", f"₹{row['avg_order_value']:,.0f}"),
            ("conv.", f"{row['avg_purchase_rate']:.1%}"),
        ]
        status_card(
            row["segment_name"],
            f"{row['user_count']:,} users · {row['pct_of_total_users']:.1f}% · {row['recommended_action']}",
            stats,
            status
        )

with col2:
    st.markdown("##### Non-buyer segments")
    for _, row in nonbuyer_segments.iterrows():
        status = status_map.get(row["segment_name"], "info")
        stats = [
            ("sessions", f"{row['avg_total_sessions']:.1f}"),
            ("duration", f"{row['avg_session_duration_sec']:.0f}s"),
            ("carts", f"{row['avg_carts']:.1f}"),
        ]
        status_card(
            row["segment_name"],
            f"{row['user_count']:,} users · {row['pct_of_total_users']:.1f}% · {row['recommended_action']}",
            stats,
            status
        )