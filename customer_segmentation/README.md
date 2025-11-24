# Customer Segmentation with K-Means + Streamlit Dashboard

## 📋 Project Overview

This is an end-to-end Machine Learning project that segments customers into distinct groups using K-Means clustering algorithm. The project includes a comprehensive Streamlit dashboard for visualization and real-time predictions.

### Key Features
- **Synthetic Dataset Generation**: Creates realistic customer data with 10 features
- **Data Preprocessing**: Handles missing values, outliers, scaling, and encoding
- **Optimal Cluster Selection**: Uses Elbow Method and Silhouette Score
- **K-Means Clustering**: Segments customers into meaningful groups
- **Interactive Dashboard**: Streamlit-based visualization and prediction interface
- **Batch Predictions**: Upload CSV files to segment multiple customers
- **FastAPI Integration**: RESTful API endpoint for cluster predictions

## 📊 Dataset Description

The dataset includes the following features:

| Feature | Description | Type |
|---------|-------------|------|
| CustomerID | Unique customer identifier | String |
| Age | Customer age (18-80) | Numerical |
| Gender | Customer gender | Categorical |
| Income | Annual income ($20K-$150K) | Numerical |
| SpendingScore | Spending behavior score (1-100) | Numerical |
| Region | Geographic region | Categorical |
| PurchaseFrequency | Number of purchases per year | Numerical |
| AvgOrderValue | Average order value ($50-$2000) | Numerical |
| Recency | Days since last purchase | Numerical |
| TotalSpend | Total annual spend (calculated) | Numerical |

## 🏗️ Project Structure

```
customer_segmentation/
│
├── data/
│   ├── generate_data.py          # Synthetic data generation script
│   ├── customers.csv              # Original dataset
│   └── customers_clustered.csv    # Dataset with cluster assignments
│
├── src/
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── clustering_model.py        # K-Means model implementation
│   └── utils.py                   # Utility functions for visualization
│
├── streamlit_app/
│   └── app.py                     # Streamlit dashboard application
│
├── model/
│   ├── kmeans_model.pkl           # Trained K-Means model
│   ├── preprocessor.pkl           # Fitted preprocessor
│   └── elbow_silhouette.png       # Model selection visualization
│
├── notebooks/
│   └── EDA_and_Training.py        # Complete training pipeline
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 How to Run Locally

### Step 1: Install Dependencies

```bash
cd /app/customer_segmentation
pip install -r requirements.txt
```

### Step 2: Generate Dataset

```bash
python data/generate_data.py
```

### Step 3: Train Model

Run the complete training pipeline:

```bash
python notebooks/EDA_and_Training.py
```

This will:
- Load and analyze the data
- Preprocess features
- Find optimal number of clusters
- Train K-Means model
- Save model and preprocessor
- Generate clustered dataset

### Step 4: Launch Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py --server.port 8501
```

Access the dashboard at: `http://localhost:8501`

## 📱 Using the Streamlit Dashboard

### Dashboard Overview
- View key metrics (total customers, clusters, average values)
- Visualize cluster distribution
- Explore scatter plots and heatmaps
- Review model performance metrics

### Cluster Analysis
- Filter specific clusters
- View detailed cluster profiles
- Compare clusters using radar charts
- Analyze feature relationships
- Export cluster statistics

### Predict New Customers
- **Single Customer**: Enter details manually for instant prediction
- **Batch Upload**: Upload CSV file to segment multiple customers
- Download predictions as CSV

### Dataset Explorer
- Browse complete dataset
- View statistical summaries
- Analyze feature correlations
- Explore distributions
- Download clustered data

## 🤖 Model Information

### Algorithm: K-Means Clustering

**Optimal Cluster Selection:**
- Elbow Method: Identifies the "elbow" in inertia curve
- Silhouette Score: Measures cluster quality (typically 2-10 clusters tested)

**Preprocessing Pipeline:**
1. Missing value imputation (median for numerical, mode for categorical)
2. Outlier detection using Z-score (threshold = 3)
3. Label encoding for categorical features
4. Standard scaling for all features

**Evaluation Metrics:**
- Inertia (within-cluster sum of squares)
- Silhouette Score (measures cluster separation)
- Davies-Bouldin Index (lower is better)

## 🔌 FastAPI Integration (Optional)

The project includes a FastAPI endpoint for cluster predictions:

### Endpoint: `/api/predict_cluster`

**Request:**
```json
POST /api/predict_cluster
{
  "age": 35,
  "gender": "Male",
  "income": 65000,
  "spending_score": 75,
  "region": "North",
  "purchase_frequency": 24,
  "avg_order_value": 450,
  "recency": 15
}
```

**Response:**
```json
{
  "cluster": 2,
  "cluster_size": 234,
  "cluster_characteristics": {
    "avg_income": 68500,
    "avg_spending_score": 72.3,
    "avg_total_spend": 11200
  }
}
```

## 📈 Example Cluster Interpretations

After training, you might get clusters like:

**Cluster 0: Budget Shoppers**
- Lower income, lower spending score
- Infrequent purchases, lower AOV
- Marketing: Focus on discounts and promotions

**Cluster 1: Premium Customers**
- High income, high spending score
- Frequent purchases, high AOV
- Marketing: Exclusive offers, loyalty programs

**Cluster 2: Occasional Buyers**
- Medium income, variable spending
- Moderate frequency, medium AOV
- Marketing: Re-engagement campaigns

*(Actual interpretations depend on your data)*

## 🧪 Testing

Test the prediction function:

```python
import pandas as pd
from src.clustering_model import CustomerSegmentation
from src.data_preprocessing import DataPreprocessor
import joblib

# Load models
model = CustomerSegmentation.load_model('model/kmeans_model.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')

# Create test customer
test_customer = pd.DataFrame({
    'Age': [35],
    'Gender': ['Male'],
    'Income': [65000],
    'SpendingScore': [75],
    'Region': ['North'],
    'PurchaseFrequency': [24],
    'AvgOrderValue': [450],
    'Recency': [15],
    'TotalSpend': [10800]
})

# Preprocess and predict
processed = preprocessor.preprocess(test_customer, remove_outliers=False, fit=False)
cluster = model.predict(processed)
print(f"Customer belongs to Cluster: {cluster[0]}")
```

## 📦 Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.14.0
- streamlit >= 1.28.0
- joblib >= 1.3.0
- fastapi >= 0.110.0 (optional)
- uvicorn >= 0.25.0 (optional)

## 🎯 Business Applications

1. **Targeted Marketing**: Customize campaigns for each segment
2. **Product Recommendations**: Suggest products based on cluster behavior
3. **Customer Retention**: Identify at-risk segments
4. **Resource Allocation**: Optimize marketing budget per segment
5. **Personalization**: Tailor user experience based on cluster

## 🚀 Future Enhancements

- [ ] Add DBSCAN and Hierarchical Clustering
- [ ] Implement PCA for dimensionality reduction
- [ ] Add customer lifetime value (CLV) prediction
- [ ] Create PDF report generator
- [ ] Add A/B testing framework
- [ ] Deploy on HuggingFace Spaces or Render
- [ ] Add authentication to dashboard
- [ ] Implement AutoML for hyperparameter tuning

## 📄 License

MIT License

## 👨‍💻 Author

Built with ❤️ using FastAPI, Streamlit, and scikit-learn

---

**Note**: This is a demonstration project using synthetic data. For production use, replace with real customer data and adjust preprocessing steps accordingly.
