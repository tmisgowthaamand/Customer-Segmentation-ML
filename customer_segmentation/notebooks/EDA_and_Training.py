#!/usr/bin/env python
# coding: utf-8

"""
Customer Segmentation - EDA and Model Training Pipeline
"""

import sys
sys.path.append('/app/customer_segmentation')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import DataPreprocessor
from src.clustering_model import CustomerSegmentation
from src.utils import get_cluster_profiles, generate_cluster_insights
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("CUSTOMER SEGMENTATION - ML PIPELINE")
print("="*80)

# 1. Load Data
print("\n[1] Loading Data...")
df = pd.read_csv('/app/customer_segmentation/data/customers.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")

# 2. Exploratory Data Analysis
print("\n[2] Exploratory Data Analysis")
print("\n--- Data Info ---")
print(df.info())
print("\n--- Statistical Summary ---")
print(df.describe())

# Missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

# Distribution of key features
print("\n--- Feature Distributions ---")
key_features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency', 'AvgOrderValue']
for feature in key_features:
    if feature in df.columns:
        print(f"{feature}: Mean={df[feature].mean():.2f}, Median={df[feature].median():.2f}, Std={df[feature].std():.2f}")

# 3. Data Preprocessing
print("\n[3] Data Preprocessing...")
preprocessor = DataPreprocessor()

# Separate features for clustering (exclude CustomerID)
features_for_clustering = df.drop('CustomerID', axis=1) if 'CustomerID' in df.columns else df

# Store original indices before preprocessing
original_indices = features_for_clustering.index

# Preprocess
df_processed = preprocessor.preprocess(features_for_clustering, remove_outliers=True, fit=True)
print(f"Processed data shape: {df_processed.shape}")

# Track which rows remained after outlier removal
retained_indices = df_processed.index

# 4. Find Optimal Number of Clusters
print("\n[4] Finding Optimal Number of Clusters...")
segmentation = CustomerSegmentation(random_state=42)
optimal_k = segmentation.find_optimal_clusters(df_processed, max_k=10, method='both')
print(f"Optimal K determined: {optimal_k}")

# Plot Elbow and Silhouette
fig = segmentation.plot_elbow_silhouette(save_path='/app/customer_segmentation/model/elbow_silhouette.png')
print("Elbow and Silhouette plots saved.")

# 5. Train K-Means Model
print("\n[5] Training K-Means Model...")
segmentation.train(df_processed)

# 6. Get Predictions
print("\n[6] Generating Cluster Assignments...")
cluster_labels = segmentation.predict(df_processed)

# Add clusters to original dataframe (only for rows that weren't removed as outliers)
df_original = pd.read_csv('/app/customer_segmentation/data/customers.csv')
df_original.loc[retained_indices, 'Cluster'] = cluster_labels

# For removed outliers, assign them to nearest cluster
if len(df_original) != len(cluster_labels):
    print(f"Assigning {len(df_original) - len(cluster_labels)} outlier rows to nearest clusters...")
    outlier_indices = df_original.index.difference(retained_indices)
    outlier_data = df_original.loc[outlier_indices]
    outlier_features = outlier_data.drop(['CustomerID', 'Cluster'], axis=1, errors='ignore')
    outlier_processed = preprocessor.preprocess(outlier_features, remove_outliers=False, fit=False)
    outlier_clusters = segmentation.predict(outlier_processed)
    df_original.loc[outlier_indices, 'Cluster'] = outlier_clusters

# Convert cluster to int
df_original['Cluster'] = df_original['Cluster'].astype(int)

# 7. Cluster Analysis
print("\n[7] Cluster Analysis")
print("\n--- Cluster Sizes ---")
print(df_original['Cluster'].value_counts().sort_index())

print("\n--- Cluster Profiles ---")
profiles = get_cluster_profiles(df_original, 'Cluster')
print(profiles)

# Generate insights
insights = generate_cluster_insights(df_original, 'Cluster')
print("\n--- Cluster Insights ---")
for cluster_id, data in insights.items():
    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {data['size']} customers ({data['percentage']:.1f}%)")
    print(f"  Avg Income: ${data['stats']['Income']['mean']:.0f}")
    print(f"  Avg Spending Score: {data['stats']['SpendingScore']['mean']:.1f}")
    print(f"  Avg Purchase Frequency: {data['stats']['PurchaseFrequency']['mean']:.1f}")

# 8. Save Model and Preprocessor
print("\n[8] Saving Model and Preprocessor...")
segmentation.save_model('/app/customer_segmentation/model/kmeans_model.pkl')
import joblib
joblib.dump(preprocessor, '/app/customer_segmentation/model/preprocessor.pkl')
print("Model and preprocessor saved successfully.")

# 9. Save Clustered Data
print("\n[9] Saving Clustered Data...")
df_original.to_csv('/app/customer_segmentation/data/customers_clustered.csv', index=False)
print("Clustered data saved.")

print("\n" + "="*80)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
