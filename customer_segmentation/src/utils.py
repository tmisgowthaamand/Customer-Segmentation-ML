import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_cluster_profiles(df, cluster_col='Cluster'):
    """
    Generate cluster profiles with statistics
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numerical_cols:
        numerical_cols.remove(cluster_col)
    
    profiles = df.groupby(cluster_col)[numerical_cols].agg(['mean', 'median', 'std', 'count'])
    return profiles

def plot_cluster_distribution(df, cluster_col='Cluster'):
    """
    Plot cluster distribution
    """
    fig = px.histogram(df, x=cluster_col, 
                       title='Customer Distribution Across Clusters',
                       labels={cluster_col: 'Cluster', 'count': 'Number of Customers'},
                       color=cluster_col,
                       text_auto=True)
    fig.update_layout(showlegend=False, height=400)
    return fig

def plot_cluster_scatter(df, x_col, y_col, cluster_col='Cluster'):
    """
    Create interactive scatter plot for clusters
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=cluster_col,
                     title=f'{y_col} vs {x_col} by Cluster',
                     labels={cluster_col: 'Cluster'},
                     hover_data=df.columns,
                     color_continuous_scale='Viridis')
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig.update_layout(height=500)
    return fig

def plot_cluster_heatmap(df, cluster_col='Cluster'):
    """
    Create heatmap of cluster characteristics
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numerical_cols:
        numerical_cols.remove(cluster_col)
    
    cluster_means = df.groupby(cluster_col)[numerical_cols].mean()
    
    # Normalize for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_means_normalized = pd.DataFrame(
        scaler.fit_transform(cluster_means.T).T,
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    fig = px.imshow(cluster_means_normalized,
                    labels=dict(x="Features", y="Cluster", color="Normalized Value"),
                    x=cluster_means_normalized.columns,
                    y=[f"Cluster {i}" for i in cluster_means_normalized.index],
                    title="Cluster Characteristics Heatmap",
                    color_continuous_scale='RdYlGn',
                    aspect='auto')
    fig.update_layout(height=400)
    return fig

def plot_radar_chart(df, cluster_col='Cluster'):
    """
    Create radar chart for cluster comparison
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numerical_cols:
        numerical_cols.remove(cluster_col)
    
    cluster_means = df.groupby(cluster_col)[numerical_cols].mean()
    
    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_means_normalized = pd.DataFrame(
        scaler.fit_transform(cluster_means.T).T,
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    fig = go.Figure()
    
    for idx in cluster_means_normalized.index:
        fig.add_trace(go.Scatterpolar(
            r=cluster_means_normalized.loc[idx].values,
            theta=cluster_means_normalized.columns,
            fill='toself',
            name=f'Cluster {idx}'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Cluster Comparison - Radar Chart",
        height=600
    )
    return fig

def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    fig.update_layout(height=600)
    return fig

def generate_cluster_insights(df, cluster_col='Cluster'):
    """
    Generate textual insights for each cluster
    """
    insights = {}
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numerical_cols:
        numerical_cols.remove(cluster_col)
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / len(df)) * 100
        
        # Calculate key statistics
        stats = {}
        for col in numerical_cols:
            stats[col] = {
                'mean': cluster_data[col].mean(),
                'median': cluster_data[col].median()
            }
        
        insights[cluster_id] = {
            'size': cluster_size,
            'percentage': cluster_pct,
            'stats': stats
        }
    
    return insights

def create_downloadable_report(df, cluster_col='Cluster'):
    """
    Create a comprehensive report DataFrame
    """
    report = df.copy()
    
    # Add cluster statistics
    profiles = get_cluster_profiles(df, cluster_col)
    
    return report, profiles
