# Customer Segmentation ML Project - Quick Start Guide

## 🚀 Quick Start

### Method 1: Launch Streamlit Dashboard (Recommended)

Open a terminal and run:
```bash
cd /app/customer_segmentation
./run_streamlit.sh
```

Then open your browser to: **http://localhost:8501**

### Method 2: Use React Frontend

The React frontend is already running. Simply open your browser to see the ML prediction interface.

### Method 3: Use FastAPI Endpoint Directly

Test the API with curl:
```bash
curl -X POST "http://localhost:8001/api/predict_cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "income": 65000,
    "spending_score": 75,
    "region": "North",
    "purchase_frequency": 24,
    "avg_order_value": 450,
    "recency": 15
  }'
```

## 📁 Project Structure

```
/app/customer_segmentation/
├── data/
│   ├── generate_data.py              # Synthetic data generator
│   ├── customers.csv                 # Original dataset (1000 customers)
│   └── customers_clustered.csv       # Dataset with cluster labels
│
├── src/
│   ├── data_preprocessing.py         # Data cleaning & preprocessing
│   ├── clustering_model.py           # K-Means model implementation
│   └── utils.py                      # Visualization utilities
│
├── streamlit_app/
│   └── app.py                        # Interactive Streamlit dashboard
│
├── model/
│   ├── kmeans_model.pkl              # Trained K-Means model
│   ├── preprocessor.pkl              # Fitted preprocessor
│   └── elbow_silhouette.png          # Model selection visualization
│
├── notebooks/
│   └── EDA_and_Training.py           # Complete training pipeline
│
├── requirements.txt                   # Python dependencies
├── run_streamlit.sh                  # Streamlit launcher script
└── README.md                         # Full documentation

/app/backend/
└── server.py                         # FastAPI with /api/predict_cluster endpoint

/app/frontend/
└── src/
    └── MLDashboard.js                # React ML prediction interface
```

## 🎯 Features Implemented

### ✅ Data Pipeline
- Synthetic customer data generation (1000 customers, 10 features)
- Missing value handling (median/mode imputation)
- Outlier detection & removal (Z-score method)
- Feature scaling (StandardScaler)
- Categorical encoding (Label encoding)

### ✅ ML Model
- K-Means clustering algorithm
- Optimal cluster selection using:
  - Elbow Method
  - Silhouette Score
- Model evaluation metrics:
  - Inertia
  - Silhouette Score: 0.257
  - Davies-Bouldin Index

### ✅ Streamlit Dashboard
- **Dashboard Overview**: Metrics, distributions, heatmaps
- **Cluster Analysis**: Detailed profiles, radar charts, comparisons
- **Predict New Customers**: Single & batch predictions
- **Dataset Explorer**: Data browsing, correlations, distributions
- CSV upload/download functionality

### ✅ FastAPI Integration
- REST endpoint: `POST /api/predict_cluster`
- Returns cluster assignment + characteristics
- Integrated with React frontend

### ✅ React Frontend
- Modern UI with Tailwind CSS & Shadcn components
- Real-time predictions
- Cluster visualization
- Marketing recommendations

## 📊 Cluster Insights

Based on the trained model:

**Cluster 0: Budget-Conscious Shoppers (55.4%)**
- Avg Income: $47,573
- Avg Spending Score: 39.5/100
- Avg Purchase Frequency: 7.6/year
- Avg Total Spend: $4,717/year
- **Strategy**: Discounts, promotions, loyalty programs

**Cluster 1: Premium Customers (44.6%)**
- Avg Income: $77,975
- Avg Spending Score: 72.5/100
- Avg Purchase Frequency: 15.1/year
- Avg Total Spend: $11,731/year
- **Strategy**: Exclusive offers, VIP programs, premium products

## 🔧 Retraining the Model

If you want to regenerate data or retrain:

```bash
cd /app/customer_segmentation

# Step 1: Generate new data
python data/generate_data.py

# Step 2: Train model
python notebooks/EDA_and_Training.py

# Step 3: Restart backend
sudo supervisorctl restart backend

# Step 4: Launch Streamlit
./run_streamlit.sh
```

## 🧪 Testing

### Test FastAPI Endpoint
```bash
# Test with premium customer profile
curl -X POST "http://localhost:8001/api/predict_cluster" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "gender": "Female", "income": 85000, "spending_score": 80, "region": "East", "purchase_frequency": 20, "avg_order_value": 600, "recency": 10}'

# Test with budget customer profile
curl -X POST "http://localhost:8001/api/predict_cluster" \
  -H "Content-Type: application/json" \
  -d '{"age": 28, "gender": "Male", "income": 35000, "spending_score": 30, "region": "West", "purchase_frequency": 5, "avg_order_value": 200, "recency": 60}'
```

## 🎨 Streamlit Dashboard Features

1. **Dashboard Overview**
   - Total customers, clusters, average metrics
   - Cluster distribution chart
   - Income vs Spending Score scatter plot
   - Cluster characteristics heatmap
   - Model performance visualization

2. **Cluster Analysis**
   - Filter by cluster
   - Detailed cluster profiles
   - Radar chart comparison
   - Feature relationship plots
   - Statistical tables

3. **Predict New Customers**
   - Single customer form (instant prediction)
   - CSV batch upload
   - Download predictions
   - Cluster distribution visualization

4. **Dataset Explorer**
   - Data preview & statistics
   - Correlation heatmap
   - Feature distributions
   - Box plots by cluster
   - Download dataset

## 📦 Dependencies

All dependencies are already installed:
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- streamlit
- fastapi, uvicorn
- joblib

## 🚀 Production Deployment

### Streamlit (Recommended)
- Deploy to Streamlit Cloud / HuggingFace Spaces
- Already production-ready

### FastAPI
- API already integrated with main backend
- Endpoint: `/api/predict_cluster`

## 📝 API Documentation

### POST /api/predict_cluster

**Request Body:**
```json
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
  "cluster": 1,
  "cluster_size": 446,
  "cluster_characteristics": {
    "avg_income": 77974.72,
    "avg_spending_score": 72.55,
    "avg_total_spend": 11731.24,
    "avg_purchase_frequency": 15.09,
    "avg_recency": 30.87
  }
}
```

## 🎯 Business Use Cases

1. **Targeted Marketing**: Customize campaigns per segment
2. **Product Recommendations**: Suggest products by cluster behavior
3. **Customer Retention**: Identify at-risk segments
4. **Resource Allocation**: Optimize marketing budget
5. **Personalization**: Tailor UX based on cluster

## 🔮 Next Steps

- Add more clusters for finer segmentation
- Implement DBSCAN or Hierarchical clustering
- Add customer lifetime value prediction
- Create PDF report generator
- Add A/B testing framework
- Integrate with CRM systems

## 📞 Support

For detailed documentation, see: `/app/customer_segmentation/README.md`

---

**Status**: ✅ Fully functional and production-ready
