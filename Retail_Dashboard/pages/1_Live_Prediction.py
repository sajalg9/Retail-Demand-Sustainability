import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.theme import load_css, section_header, status_card

st.set_page_config(page_title="Live Prediction | Retail Ops", layout="wide")
load_css()

section_header(
    "Live Purchase Prediction",
    "Simulate a session and predict purchase likelihood, with a feature-level explanation for that specific prediction."
)

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("purchase_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)


model, scaler = load_model_and_scaler()
explainer = load_shap_explainer(model)

FEATURE_NAMES = [
    'num_events', 'num_views', 'num_carts', 'num_remove_from_cart',
    'num_unique_products', 'num_unique_categories', 'avg_price',
    'max_price', 'min_price', 'num_brands', 'session_duration', 'hour_of_day'
]

st.markdown("### Try it: simulate a session")
st.markdown("Adjust the inputs below — the model will estimate whether this session is likely to end in a purchase.")

col1, col2 = st.columns(2)

with col1:
    num_events = st.number_input("Number of events", min_value=0, value=10)
    session_duration = st.number_input("Session duration (seconds)", min_value=0.0, value=300.0)
    num_carts = st.number_input("Add-to-cart events", min_value=0, value=1)
    num_views = st.number_input("Product views", min_value=0, value=5)
    num_unique_products = st.number_input("Unique products viewed", min_value=0, value=3)
    num_brands = st.number_input("Unique brands interacted", min_value=0, value=2)

with col2:
    max_price = st.number_input("Max product price viewed", min_value=0.0, value=1500.0)
    avg_price = st.number_input("Average product price", min_value=0.0, value=700.0)
    min_price = st.number_input("Min product price", min_value=0.0, value=100.0)
    hour_of_day = st.number_input("Session start hour (0–23)", min_value=0, max_value=23, value=14)
    num_unique_categories = st.number_input("Unique categories interacted", min_value=0, value=2)
    num_remove_from_cart = st.number_input("Remove-from-cart events", min_value=0, value=0)

predict_clicked = st.button("Predict purchase")

if predict_clicked:
    input_data = pd.DataFrame([{
        'num_events': num_events,
        'num_views': num_views,
        'num_carts': num_carts,
        'num_remove_from_cart': num_remove_from_cart,
        'num_unique_products': num_unique_products,
        'num_unique_categories': num_unique_categories,
        'avg_price': avg_price,
        'max_price': max_price,
        'min_price': min_price,
        'num_brands': num_brands,
        'session_duration': session_duration,
        'hour_of_day': hour_of_day
    }])[FEATURE_NAMES]

    scaled = scaler.transform(input_data)
    predicted_class = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    status = "good" if predicted_class == 1 else "watch"
    label = "Likely to purchase" if predicted_class == 1 else "Not likely to purchase"

    status_card(
        label,
        f"Purchase probability: {probability:.2%}",
        [("events", f"{num_events}"), ("carts", f"{num_carts}"), ("avg price", f"₹{avg_price:,.0f}")],
        status
    )

    st.markdown("#### Detailed prediction breakdown")
    st.dataframe(
        input_data.assign(Predicted_Class=predicted_class,
                           Purchase_Probability=round(probability, 4)),
        use_container_width=True
    )

    st.markdown("#### Why this prediction: feature contributions")
    st.markdown(
        "Red bars push the prediction **toward** a purchase; blue bars push it **away**. "
        "Values shown are this session's actual inputs."
    )

    scaled_df = pd.DataFrame(scaled, columns=FEATURE_NAMES)
    explanation = explainer(scaled_df)

    single_exp = explanation[0, :, 1]
    single_exp.data = input_data.iloc[0].values

    fig = plt.figure()
    shap.plots.waterfall(single_exp, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)