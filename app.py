import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Customer Intelligence Suite", layout="wide", page_icon="📊")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analytics Dashboard", "Customer Segment Predictor"])

# --- SHARED DATA LOADING ---
@st.cache_resource
def load_models():
    # Loading the KMeans model and RobustScaler
    model = joblib.load('customer_segmentation_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# --- PAGE 1: ANALYTICS DASHBOARD ---
if page == "Analytics Dashboard":
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("Detailed breakdown of identified customer clusters and business impact.")

    try:
        # Load the exported Excel data
        cluster_summary = pd.read_excel("cluster_summary.xlsx")
        persona_table = pd.read_excel("cluster_personas.xlsx")
        education_group_dist = pd.read_excel("cluster_education_group_distribution.xlsx", index_col=0)
        marital_group_dist = pd.read_excel("cluster_marital_group_distribution.xlsx", index_col=0)

        # 1. Cluster Personas Table
        st.header("1. Cluster Personas and Business Actions")
        st.dataframe(persona_table, use_container_width=True)

        # 2. Revenue & Distribution Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Customer Distribution")
            fig_cust_dist = px.bar(cluster_summary, x='Cluster_Name', y='Customers', color='Cluster_Name')
            st.plotly_chart(fig_cust_dist, use_container_width=True)
        
        with col2:
            st.subheader("Estimated Revenue Contribution")
            rev_contribution = cluster_summary.copy()
            rev_contribution['Total_Revenue_Proxy'] = rev_contribution['Avg_Total_Spend'] * rev_contribution['Customer_%']
            fig_rev_dist = px.pie(rev_contribution, values='Total_Revenue_Proxy', names='Cluster_Name')
            st.plotly_chart(fig_rev_dist, use_container_width=True)

        # 3. Demographic Heatmaps
        st.header("2. Demographic Heatmaps")
        tab1, tab2 = st.tabs(["Marital Status", "Education Level"])
        with tab1:
            fig_marital, ax_mar = plt.subplots(figsize=(10, 5))
            sns.heatmap(marital_group_dist, annot=True, cmap="YlGnBu", fmt=".1%", ax=ax_mar)
            st.pyplot(fig_marital)
        with tab2:
            fig_edu, ax_edu = plt.subplots(figsize=(10, 5))
            sns.heatmap(education_group_dist, annot=True, cmap="YlGnBu", fmt=".1%", ax=ax_edu)
            st.pyplot(fig_edu)

    except Exception as e:
        st.error(f"Error loading dashboard data: {e}. Ensure all Excel files are uploaded.")

# --- PAGE 2: CUSTOMER PREDICTOR ---
elif page == "Customer Segment Predictor":
    st.title("🛍 Customer Segment Predictor")
    st.write("Enter specific customer details to predict their segment and strategy.")

    model, scaler = load_models()

    # User Inputs for the two primary features
    col_a, col_b = st.columns(2)
    with col_a:
        income = st.number_input('Annual Income (k$)', min_value=0, value=50)
    with col_b:
        spending = st.number_input('Spending Score (1–100)', min_value=1, max_value=100, value=50)

    if st.button('Predict Segment', use_container_width=True):
        # Using RobustScaler 'center_' (medians) as baseline for the features
        input_features = scaler.center_.copy().reshape(1, -1)

        # Map user inputs (Income: index 3, Total Spend: index 28)
        input_features[0, 3] = income 
        input_features[0, 28] = spending 

        # Scale and predict
        features_scaled = scaler.transform(input_features)
        cluster = model.predict(features_scaled)[0]

        st.divider()
        st.subheader(f"🎯 Prediction: Segment {cluster}")

        # Visualization of customer vs cluster centers
        centers = scaler.inverse_transform(model.cluster_centers_)
        fig_map, ax_map = plt.subplots()
        ax_map.scatter(centers[:, 3], centers[:, 28], c='blue', s=200, marker='X', label='Segment Centers')
        ax_map.scatter(income, spending, c='red', s=300, label='This Customer', edgecolors='white')
        ax_map.set_xlabel("Income")
        ax_map.set_ylabel("Spending Score")
        ax_map.legend()
        st.pyplot(fig_map)
        
        st.balloons()
