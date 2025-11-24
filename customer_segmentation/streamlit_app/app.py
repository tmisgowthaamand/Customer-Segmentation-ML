import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append('/app/customer_segmentation')

from src.data_preprocessing import DataPreprocessor
from src.clustering_model import CustomerSegmentation
from src.utils import (
    get_cluster_profiles,
    plot_cluster_distribution,
    plot_cluster_scatter,
    plot_cluster_heatmap,
    plot_radar_chart,
    plot_correlation_heatmap,
    generate_cluster_insights
)
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cluster-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_models():
    try:
        model = CustomerSegmentation.load_model('/app/customer_segmentation/model/kmeans_model.pkl')
        preprocessor = joblib.load('/app/customer_segmentation/model/preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load clustered data
@st.cache_data
def load_clustered_data():
    try:
        df = pd.read_csv('/app/customer_segmentation/data/customers_clustered.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">📊 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem;">AI-Powered Customer Insights with K-Means Clustering</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard Overview", "Cluster Analysis", "Predict New Customers", "Dataset Explorer"]
    )
    
    # Load data and models
    df = load_clustered_data()
    model, preprocessor = load_models()
    
    if df is None:
        st.error("⚠️ Please run the training pipeline first to generate clustered data.")
        st.code("python /app/customer_segmentation/notebooks/EDA_and_Training.py")
        return
    
    # Page routing
    if page == "Dashboard Overview":
        show_dashboard(df, model)
    elif page == "Cluster Analysis":
        show_cluster_analysis(df)
    elif page == "Predict New Customers":
        show_prediction_page(model, preprocessor)
    elif page == "Dataset Explorer":
        show_dataset_explorer(df)

def show_dashboard(df, model):
    st.markdown('<h2 class="sub-header">📈 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Number of Clusters", f"{df['Cluster'].nunique()}")
    with col3:
        st.metric("Avg Customer Value", f"${df['TotalSpend'].mean():,.0f}")
    with col4:
        st.metric("Avg Purchase Frequency", f"{df['PurchaseFrequency'].mean():.1f}")
    
    st.markdown("---")
    
    # Cluster Distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">Cluster Distribution</h3>', unsafe_allow_html=True)
        fig = plot_cluster_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">Income vs Spending Score</h3>', unsafe_allow_html=True)
        fig = plot_cluster_scatter(df, 'Income', 'SpendingScore')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Characteristics
    st.markdown('<h3 class="sub-header">Cluster Characteristics Heatmap</h3>', unsafe_allow_html=True)
    fig = plot_cluster_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance
    if model:
        st.markdown('<h3 class="sub-header">Model Performance Metrics</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                from PIL import Image
                img = Image.open('/app/customer_segmentation/model/elbow_silhouette.png')
                st.image(img, caption='Elbow Method & Silhouette Analysis', use_container_width=True)
            except:
                st.info("Elbow/Silhouette plot not available")
        
        with col2:
            st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
            st.markdown("**Model Details**")
            st.write(f"- Algorithm: K-Means Clustering")
            st.write(f"- Number of Clusters: {model.n_clusters}")
            st.write(f"- Optimal K: {model.optimal_k}")
            if model.model:
                st.write(f"- Inertia: {model.model.inertia_:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

def show_cluster_analysis(df):
    st.markdown('<h2 class="sub-header">🔍 Cluster Analysis</h2>', unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("### Filters")
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters",
        options=sorted(df['Cluster'].unique()),
        default=sorted(df['Cluster'].unique())
    )
    
    if not selected_clusters:
        st.warning("Please select at least one cluster")
        return
    
    df_filtered = df[df['Cluster'].isin(selected_clusters)]
    
    # Cluster profiles
    st.markdown('<h3 class="sub-header">Cluster Profiles</h3>', unsafe_allow_html=True)
    
    insights = generate_cluster_insights(df_filtered)
    
    # Display cluster cards
    cols = st.columns(min(len(selected_clusters), 3))
    for idx, (cluster_id, data) in enumerate(insights.items()):
        with cols[idx % 3]:
            st.markdown(f'<div class="cluster-card">', unsafe_allow_html=True)
            st.markdown(f"### Cluster {cluster_id}")
            st.write(f"**Size:** {data['size']} ({data['percentage']:.1f}%)")
            st.write(f"**Avg Income:** ${data['stats']['Income']['mean']:,.0f}")
            st.write(f"**Avg Spending Score:** {data['stats']['SpendingScore']['mean']:.1f}")
            st.write(f"**Avg Total Spend:** ${data['stats']['TotalSpend']['mean']:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Radar chart
    st.markdown('<h3 class="sub-header">Cluster Comparison - Radar Chart</h3>', unsafe_allow_html=True)
    fig = plot_radar_chart(df_filtered)
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    st.markdown('<h3 class="sub-header">Feature Relationships</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-axis", ['Income', 'Age', 'SpendingScore', 'PurchaseFrequency', 'AvgOrderValue', 'Recency'], key='x1')
    with col2:
        y_feature = st.selectbox("Y-axis", ['SpendingScore', 'TotalSpend', 'Income', 'PurchaseFrequency', 'AvgOrderValue'], key='y1')
    
    fig = plot_cluster_scatter(df_filtered, x_feature, y_feature)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed statistics
    st.markdown('<h3 class="sub-header">Detailed Cluster Statistics</h3>', unsafe_allow_html=True)
    profiles = get_cluster_profiles(df_filtered)
    st.dataframe(profiles, use_container_width=True)

def show_prediction_page(model, preprocessor):
    st.markdown('<h2 class="sub-header">🎯 Predict New Customers</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Single Customer", "Batch Upload"])
    
    with tab1:
        st.markdown("### Enter Customer Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            income = st.number_input("Annual Income ($)", min_value=20000, max_value=200000, value=50000, step=5000)
        
        with col2:
            spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
            purchase_freq = st.number_input("Purchase Frequency (per year)", min_value=1, max_value=100, value=12)
        
        with col3:
            aov = st.number_input("Average Order Value ($)", min_value=50, max_value=5000, value=500, step=50)
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
            total_spend = purchase_freq * aov
            st.metric("Calculated Total Spend", f"${total_spend:,.0f}")
        
        if st.button("🔮 Predict Cluster", key="predict_single"):
            if model and preprocessor:
                try:
                    # Create dataframe
                    customer_data = pd.DataFrame({
                        'Age': [age],
                        'Gender': [gender],
                        'Income': [income],
                        'SpendingScore': [spending_score],
                        'Region': [region],
                        'PurchaseFrequency': [purchase_freq],
                        'AvgOrderValue': [aov],
                        'Recency': [recency],
                        'TotalSpend': [total_spend]
                    })
                    
                    # Preprocess
                    customer_processed = preprocessor.preprocess(customer_data, remove_outliers=False, fit=False)
                    
                    # Predict
                    cluster = model.predict(customer_processed)[0]
                    
                    st.success(f"### Customer belongs to Cluster {cluster}")
                    
                    # Load reference data for comparison
                    df_ref = load_clustered_data()
                    cluster_data = df_ref[df_ref['Cluster'] == cluster]
                    
                    st.markdown("#### Cluster Characteristics:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Cluster Size", f"{len(cluster_data)} customers")
                    with col2:
                        st.metric("Avg Income in Cluster", f"${cluster_data['Income'].mean():,.0f}")
                    with col3:
                        st.metric("Avg Spending Score", f"{cluster_data['SpendingScore'].mean():.1f}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.error("Model not loaded. Please train the model first.")
    
    with tab2:
        st.markdown("### Upload CSV for Batch Predictions")
        st.info("Upload a CSV file with columns: Age, Gender, Income, SpendingScore, Region, PurchaseFrequency, AvgOrderValue, Recency")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df_upload.head())
                
                if st.button("🔮 Predict Clusters", key="predict_batch"):
                    if model and preprocessor:
                        # Calculate TotalSpend if not present
                        if 'TotalSpend' not in df_upload.columns:
                            df_upload['TotalSpend'] = df_upload['PurchaseFrequency'] * df_upload['AvgOrderValue']
                        
                        # Preprocess
                        df_processed = preprocessor.preprocess(df_upload.copy(), remove_outliers=False, fit=False)
                        
                        # Predict
                        if 'CustomerID' in df_processed.columns:
                            df_features = df_processed.drop('CustomerID', axis=1)
                        else:
                            df_features = df_processed
                        
                        clusters = model.predict(df_features)
                        df_upload['Cluster'] = clusters
                        
                        st.success("✅ Predictions completed!")
                        st.dataframe(df_upload)
                        
                        # Download button
                        csv = df_upload.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="customer_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Show distribution
                        st.markdown("### Cluster Distribution")
                        fig = plot_cluster_distribution(df_upload)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Model not loaded.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

def show_dataset_explorer(df):
    st.markdown('<h2 class="sub-header">📊 Dataset Explorer</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Statistical summary
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### Feature Correlations")
    fig = plot_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.markdown("### Feature Distributions")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Cluster' in numerical_cols:
        numerical_cols.remove('Cluster')
    
    selected_feature = st.selectbox("Select Feature", numerical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=selected_feature, color='Cluster', title=f"{selected_feature} by Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    # Download data
    st.markdown("### Download Dataset")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Dataset",
        data=csv,
        file_name="customers_clustered.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
