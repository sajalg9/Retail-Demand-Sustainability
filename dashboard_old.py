import streamlit as st
from PIL import Image
import os
import pandas as pd
import joblib

# --- Set page config ---
st.set_page_config(page_title="Retail Demand & Sustainability", layout="wide")

# --- Custom styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .reportview-container {
            padding-top: 1rem;
        }
        .block-container {
            padding: 2rem 2rem;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

# --- Image directory ---
img_dir = "Visualizations"  # Update if different

# --- Image loading helper ---
def show_image(filename, caption="", use_container_width=True):
    path = os.path.join(img_dir, filename)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=use_container_width)
    else:
        st.warning(f"⚠ {filename} not found in {img_dir}")

# --- Sidebar Navigation ---
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Live Purchase Classifier",
    "Purchase Prediction",
    "Demand Forecasting",
    "Sustainability Analysis"
])

# --- Overview ---
if page == "Overview":
    st.title("📦 Retail Demand & Sustainability")
    st.markdown("""
    📊 Welcome to our Retail Intelligence Dashboard

This interactive dashboard presents a full-stack, machine learning–powered solution that addresses key challenges in modern retail operations. It is the result of an end-to-end data-driven pipeline encompassing:

- **Purchase Behaviour Prediction** : Leveraging session-level features (time spent, user activity, etc.), a classification model (Random Forest) was trained to predict whether a user will complete a purchase during a session. The model achieved high accuracy and interpretability through feature importance visualizations, ROC curves.

- **Platform-Level Demand Forecasting** : Using historical daily demand data (Oct–Nov), we trained a Prophet time series model to forecast demand for the month of December. The forecasts help anticipate peak activity periods and guide capacity planning. Evaluation metrics (MAE, RMSE) validate the model's performance.

- **Sustainability-Driven Category Analysis** : Demand was segregated by product category to uncover top and bottom performers. Forecasts were compared against simulated supply limits to detect understock/overstock risks. Alert systems were implemented for decision support. This enables actionable recommendations for inventory optimization and sustainable category management.

- **Visualizations & Insights** : A range of analytical visualizations are provided to support business decision-making.

🛠 Built with Python, Scikit-Learn, Prophet, Pandas, Matplotlib, Seaborn, and Streamlit.
    """)
    st.subheader("✨ Project Summary")
    show_image("Overview.png")

# --- Model 
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("purchase_model.pkl") 
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()
if page == "Live Purchase Classifier":
    st.title("🕐 Real-Time Purchase Prediction")
    st.markdown("""
    This tool lets you simulate a user session and predict purchase likelihood.
    Adjust the input features given below, and the model will tell you if a purchase is expected.
    """)

    col1, col2 = st.columns(2)

    with col1:
        num_events = st.number_input("🧾 Number of Events", min_value=0, value=10)
        session_duration = st.number_input("⏱️ Session Duration (seconds)", min_value=0.0, value=300.0)
        num_carts = st.number_input("🛒 Add-to-Cart Events", min_value=0, value=1)
        num_views = st.number_input("👀 Product Views", min_value=0, value=5)
        num_unique_products = st.number_input("🔁 Unique Products Viewed", min_value=0, value=3)
        num_brands = st.number_input("🏷️ Unique Brands Interacted", min_value=0, value=2)

    with col2:
        max_price = st.number_input("💰 Max Product Price Viewed", min_value=0.0, value=1500.0)
        avg_price = st.number_input("💵 Average Product Price", min_value=0.0, value=700.0)
        min_price = st.number_input("🪙 Min Product Price", min_value=0.0, value=100.0)
        hour_of_day = st.number_input("🕐 Session Start Hour (0–23)", min_value=0, max_value=23, value=14)
        num_unique_categories = st.number_input("📦 Unique Categories Interacted", min_value=0, value=2)
        num_remove_from_cart = st.number_input("🗑️ Remove-from-Cart Events", min_value=0, value=0)

    if st.button("🔍 Predict Purchase"):
        input_data = pd.DataFrame([{
            'num_events' : num_events, 
            'num_views' : num_views,
            'num_carts' : num_carts,
            'num_remove_from_cart' : num_remove_from_cart,
            'num_unique_products' : num_unique_products,
            'num_unique_categories' : num_unique_categories,
            'avg_price' : avg_price, 
            'max_price' : max_price, 
            'min_price' : min_price,
            'num_brands' : num_brands, 
            'session_duration' : session_duration,
            'hour_of_day' : hour_of_day
        }])

        scaled = scaler.transform(input_data)
        predicted_class = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        if predicted_class == 1:
            st.success(f"✅ Likely to Purchase (Purchase Probability: {probability:.2%})")
        else:
            st.warning(f"❌ Not Likely to Purchase (Purchase Probability: {probability:.2%})")

        st.markdown("#### 🔍 Detailed Prediction Breakdown")
        st.dataframe(input_data.assign(Predicted_Class=predicted_class,
                                       Purchase_Probability=round(probability, 4)))




# --- Phase 3A ---
elif page == "Purchase Prediction":
    st.title("🧠 Purchase Classification")
    st.markdown("Using a trained classification model to predict if a user session will lead to a purchase.")
    st.subheader("Model Summary: Purchase Classifier")

    st.markdown("""
    - *Model*: Random Forest Classifier  

    #### 📊 Evaluation Metrics (Class-wise)
    """)
    
    # Display metrics in table format
    metrics_df = pd.DataFrame({
        "Class": ["0 (No Purchase)", "1 (Purchase)"],
        "Precision": ["1.00", "0.57"],
        "Recall": ["0.95", "0.95"],
        "F1-score": ["0.97", "0.71"]
    })
    metrics_df.index = metrics_df.index + 1
    st.dataframe(metrics_df)
    
    # Accuracy display
    st.metric(label="✅ Overall Accuracy", value="94.80%")
    
    st.markdown("""
    - This model is highly precise and reliable at detecting non-purchase behavior, and balances good recall for predicting actual purchases.
    - The high F1-score for class 0 reflects excellent precision and recall, while for class 1, high recall supports decision-making for targeting purchase intents.
    """)
    col1, col2 = st.columns(2)
    with col1:
        show_image("3A_Feature Importance.png")
    with col2:
        show_image("confusion_matrix.png")

    col3, col4 = st.columns(2)
    with col3:
        show_image("ROC Curve.png")
    with col4:
        show_image("Precision-Recall Curve.png")

    st.subheader("🔍 Session Distribution")
    show_image("hourly_activity_distribution.png")

# --- Phase 3B ---
elif page == "Demand Forecasting":
    st.title("📈 Demand Forecasting")
    st.markdown("Forecasting daily purchase trends to anticipate demand and plan operations.")
    st.subheader("Model Summary: Demand Forecasting")
    st.markdown(""" 
    - **Input**: Daily demand data (Oct–Nov)  
    - **Forecast Period**: December
    
    #### 📊 Evaluation Metrics
    """)
    
    metrics_3b = {
        "Mean Absolute Error (MAE)": "14,470.99",
        "Root Mean Squared Error (RMSE)": "29,962.52"
    }
    df = pd.DataFrame(metrics_3b.items(), columns=["Metric", "Value"])
    df.index = df.index + 1  # Start index from 1
    st.dataframe(df)
    
    st.markdown("""
    🔎 These metrics indicate the model's average and squared error over the test period. Prophet effectively captures seasonality and trends in retail demand data.
    """)
    st.subheader("Platform Daily Purchase Forecast: Oct–Nov Actuals + Dec Forecast")
    show_image("phase3b_forecast_plot_with_dec_line_full.png")
    col1, col2 = st.columns(2)
    with col1:
        show_image("Demand Trend October–November 2019.png")
    with col2:
        show_image("Cumulative Predicted Purchases – December.png")


# --- Phase 4 ---
elif page == "Sustainability Analysis":
    st.title("🌿 Sustainability & Inventory Insights")
    st.markdown("Category-level forecasting to identify under/overstock conditions.")

    col1, col2 = st.columns(2)
    with col1:
        show_image(r"Top_Bottom_Categories_BarChart.png")
    with col2:
        show_image(r"Number of Categories.png")

    st.subheader("📆 Weekly Demand Stackplot")
    show_image("stacked_area_plot_top5.png")
    

