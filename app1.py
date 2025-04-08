import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Big Mart Sales Predictor",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern and professional UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .title {
        font-size: 42px;
        color: #1e3a8a;
        text-align: center;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #e6f3ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 26px;
        color: #1e3a8a;
        border: 2px solid #007bff;
        margin-top: 20px;
    }
    .input-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stSlider>div>div>div>input {
        background-color: #e9ecef;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model with caching
@st.cache_resource
def load_model():
    model_path = "Train.pkl"
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file 'Train.pkl' not found.")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/711/711284.png", width=120)
    st.header("Big Mart Sales Predictor")
    st.write("Use this tool to predict sales for a specific outlet.")
    st.markdown("### How to Use")
    st.write("- Enter item and outlet details.")
    st.write("- Click 'Predict Sales' to view the result.")
    st.write("- Download the prediction as CSV.")
    if st.button("Reset Inputs"):
        st.session_state.clear()

# Main content
st.markdown('<p class="title">Big Mart Sales Prediction 🛍️</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Accurate Sales Forecasting Made Simple</p>', unsafe_allow_html=True)

# Input Section
st.markdown("### Enter Prediction Details")
col1, col2 = st.columns(2)

# Input fields in Column 1
with col1:
    ##st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("Item Details")
    item_weight = st.slider("Item Weight (kg)", 4.0, 22.0, 12.0, 0.1, key="item_weight")
    item_visibility = st.slider("Item Visibility (%)", 0.0, 0.33, 0.05, 0.01, format="%.3f", key="item_visibility")
    item_mrp = st.slider("Item MRP ($)", 30.0, 270.0, 150.0, 1.0, key="item_mrp")
    st.markdown('</div>', unsafe_allow_html=True)

# Input fields in Column 2
with col2:
    ##st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("Outlet Details")
    outlet_year = st.slider("Outlet Establishment Year", 1985, 2009, 1999, 1, key="outlet_year")
    outlet_size = st.selectbox("Outlet Size", options=["Small", "Medium", "High"], key="outlet_size")
    outlet_type = st.selectbox("Outlet Type", options=["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"], key="outlet_type")
    st.markdown('</div>', unsafe_allow_html=True)

# Encoding categorical variables
outlet_size_map = {"Small": 0, "Medium": 1, "High": 2}
outlet_type_map = {"Grocery Store": 0, "Supermarket Type1": 1, "Supermarket Type2": 2, "Supermarket Type3": 3}

outlet_size_encoded = outlet_size_map[outlet_size]
outlet_type_encoded = outlet_type_map[outlet_type]

# Predict button and results
if st.button("Predict Sales"):
    if model is None:
        st.error("Model not loaded. Please ensure 'Train.pkl' is available.")
    else:
        with st.spinner("Calculating prediction..."):
            # Single prediction
            input_data = np.array([[item_weight, item_visibility, item_mrp, outlet_year, outlet_size_encoded, outlet_type_encoded]])
            single_prediction = model.predict(input_data)[0]
            st.markdown(f'<div class="prediction-box">Predicted Sales for {outlet_year}: <b>${single_prediction:.2f}</b></div>', unsafe_allow_html=True)

            # Downloadable CSV for single prediction
            df_prediction = pd.DataFrame({
                "Item_Weight": [item_weight],
                "Item_Visibility": [item_visibility],
                "Item_MRP": [item_mrp],
                "Outlet_Establishment_Year": [outlet_year],
                "Outlet_Size": [outlet_size],
                "Outlet_Type": [outlet_type],
                "Predicted_Sales": [single_prediction]
            })
            
            st.write("")
            csv = df_prediction.to_csv(index=False)
            st.download_button(
                label="Download Prediction as CSV",
                data=csv,
                file_name=f"big_mart_sales_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            st.success("Prediction complete!")
            

# Footer
st.markdown("""
    <hr style="border: 1px solid #e2e8f0;">
    <p style='text-align: center; color: #64748b; font-size: 14px;'>Built by Nirali and Nirjara</p>
""", unsafe_allow_html=True)