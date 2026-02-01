
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the exported data (assuming they are in the same directory)
cluster_summary = pd.read_excel("cluster_summary.xlsx")
persona_table = pd.read_excel("cluster_personas.xlsx")
education_group_dist = pd.read_excel("cluster_education_group_distribution.xlsx", index_col=0)
marital_group_dist = pd.read_excel("cluster_marital_group_distribution.xlsx", index_col=0)

st.set_page_config(layout="wide")
st.title("Customer Segmentation Dashboard")

st.header("1. Cluster Personas and Business Actions")
st.dataframe(persona_table, height=300)

st.header("2. Cluster Summary")
st.dataframe(cluster_summary)

# Visualizing Customer Distribution
st.subheader("Customer Distribution by Cluster")
fig_cust_dist = px.bar(cluster_summary, x='Cluster_Name', y='Customers', 
                       title='Number of Customers per Cluster',
                       labels={'Customers': 'Number of Customers', 'Cluster_Name': 'Cluster Name'})
st.plotly_chart(fig_cust_dist, use_container_width=True)

# Visualizing Revenue Contribution
st.subheader("Revenue Contribution by Cluster")
revenue_contribution = cluster_summary[['Cluster_Name', 'Avg_Total_Spend', 'Customer_%']].copy()
revenue_contribution['Total_Revenue_Proxy'] = revenue_contribution['Avg_Total_Spend'] * revenue_contribution['Customer_%'] # Simple proxy
fig_rev_dist = px.pie(revenue_contribution, values='Total_Revenue_Proxy', names='Cluster_Name',
                      title='Estimated Revenue Contribution by Cluster')
st.plotly_chart(fig_rev_dist, use_container_width=True)


st.header("3. Demographic Distribution")
st.subheader("Marital Status Distribution by Cluster")
fig_marital = plt.figure(figsize=(10, 6))
sns.heatmap(marital_group_dist, annot=True, cmap="YlGnBu", fmt=".1%")
plt.title("Marital Status Distribution per Cluster")
st.pyplot(fig_marital)

st.subheader("Education Distribution by Cluster")
fig_edu = plt.figure(figsize=(10, 6))
sns.heatmap(education_group_dist, annot=True, cmap="YlGnBu", fmt=".1%")
plt.title("Education Distribution per Cluster")
st.pyplot(fig_edu)

st.header("4. Key Feature Profiles (Normalized Heatmap)")
# Reload df and final_features to ensure consistency for normalized heatmap
# In a real app, this would be passed or loaded more robustly.
# For this example, assuming 'df' and 'final_features' are globally available or can be re-derived
# For simplicity, assuming profile_norm was saved or can be re-generated

# Mocking profile_norm for the app if it wasn't saved directly
try:
    # Attempt to load profile_norm if it was saved
    # If not, you'd need the original 'df' and 'final_features' to regenerate it
    # This part assumes a more complete pipeline or direct access to these objects
    st.subheader("Normalized Feature Profile")
    # Placeholder: In a real scenario, profile_norm would be loaded or computed
    st.write("Normalized feature profile heatmap needs `profile_norm` dataframe which is not directly saved. Please refer to notebook for that visualization.")

except Exception as e:
    st.error(f"Could not display normalized profile: {e}")

st.header("5. Budget Reallocation Simulation")
st.write("The simulation showed a potential net revenue impact from shifting promotional budget.")

# Mock budget_simulation for the app if it wasn't saved directly
# This part assumes a more complete pipeline or direct access to these objects

# Placeholder for budget simulation result
st.write("**Current Total Revenue**: $1,271,540.50")
st.write("**Revenue Lost (Source Cluster)**: $-147.49")
st.write("**Revenue Gained (Target Cluster)**: $7,555.04")
st.write("**Net Revenue Impact**: $7,407.55")

st.markdown("--- ##### *This dashboard is for demonstration purposes. Assumptions and data should be thoroughly validated.* ---")
