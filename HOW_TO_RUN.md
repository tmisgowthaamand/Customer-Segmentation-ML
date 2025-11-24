# 🚀 How to Run - Customer Segmentation ML Project

## Quick Access Options

### Option 1: React Web Interface (FASTEST - Already Running!)
**URL**: http://localhost:3000

Just open your browser and start predicting! The React frontend is already running.

Features:
- Beautiful modern UI
- Single customer predictions
- Real-time results
- Marketing recommendations
- No setup needed

---

### Option 2: Streamlit Dashboard (MOST FEATURES)
**Launch Command**:
```bash
cd /app/customer_segmentation
./run_streamlit.sh
```

**Access**: http://localhost:8501

Features:
- 4-page comprehensive dashboard
- Batch CSV upload
- Advanced visualizations
- Cluster analysis
- Dataset explorer
- Download results

**Alternative Launch**:
```bash
streamlit run /app/customer_segmentation/streamlit_app/app.py --server.port 8501
```

---

### Option 3: API Endpoint (FOR DEVELOPERS)
The FastAPI endpoint is already running at: http://localhost:8001

**Test with curl**:
```bash
# Premium Customer Example
curl -X POST "http://localhost:8001/api/predict_cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "Female",
    "income": 85000,
    "spending_score": 80,
    "region": "East",
    "purchase_frequency": 20,
    "avg_order_value": 600,
    "recency": 10
  }'

# Budget Customer Example
curl -X POST "http://localhost:8001/api/predict_cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "gender": "Male",
    "income": 35000,
    "spending_score": 30,
    "region": "West",
    "purchase_frequency": 5,
    "avg_order_value": 200,
    "recency": 60
  }'
```

**Expected Response**:
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

---

## 🔧 Retraining the Model

If you want to generate new data or retrain with different parameters:

### Step 1: Generate New Dataset
```bash
cd /app/customer_segmentation
python data/generate_data.py
```

### Step 2: Train Model
```bash
python notebooks/EDA_and_Training.py
```

This will:
- Perform EDA
- Find optimal clusters
- Train K-Means model
- Save model & preprocessor
- Generate clustered dataset
- Create visualizations

### Step 3: Restart Backend (if needed)
```bash
sudo supervisorctl restart backend
```

### Step 4: Launch Streamlit
```bash
./run_streamlit.sh
```

---

## 📁 Important File Locations

### Models
- Trained Model: `/app/customer_segmentation/model/kmeans_model.pkl`
- Preprocessor: `/app/customer_segmentation/model/preprocessor.pkl`
- Visualization: `/app/customer_segmentation/model/elbow_silhouette.png`

### Data
- Original Data: `/app/customer_segmentation/data/customers.csv`
- Clustered Data: `/app/customer_segmentation/data/customers_clustered.csv`

### Code
- Streamlit App: `/app/customer_segmentation/streamlit_app/app.py`
- React Frontend: `/app/frontend/src/MLDashboard.js`
- FastAPI Endpoint: `/app/backend/server.py`
- ML Pipeline: `/app/customer_segmentation/notebooks/EDA_and_Training.py`

### Documentation
- Full Docs: `/app/customer_segmentation/README.md`
- Quick Start: `/app/QUICKSTART.md`
- Summary: `/app/PROJECT_SUMMARY.md`
- This Guide: `/app/HOW_TO_RUN.md`

---

## 🎯 What Each Interface Offers

### React Frontend (http://localhost:3000)
✅ Modern, beautiful UI
✅ Single customer prediction
✅ Real-time results
✅ Cluster characteristics
✅ Marketing recommendations
✅ No setup needed
❌ No batch upload
❌ Limited analytics

**Best for**: Quick predictions, demos, client-facing

### Streamlit Dashboard (http://localhost:8501)
✅ Comprehensive analytics
✅ Batch CSV upload
✅ Advanced visualizations
✅ Cluster analysis tools
✅ Dataset explorer
✅ Download results
❌ Requires separate launch

**Best for**: Deep analysis, data exploration, batch processing

### API Endpoint (http://localhost:8001/api/predict_cluster)
✅ Programmatic access
✅ Integration-ready
✅ JSON responses
✅ Fast & lightweight
❌ No UI
❌ Requires API knowledge

**Best for**: System integration, automation, mobile apps

---

## 🧪 Testing

### Test React Frontend
1. Open http://localhost:3000
2. Enter customer details
3. Click "Predict Customer Segment"
4. Verify results appear

### Test Streamlit
1. Run `./run_streamlit.sh`
2. Open http://localhost:8501
3. Navigate through all 4 pages
4. Try uploading a CSV

### Test API
```bash
# Quick test
curl -X POST "http://localhost:8001/api/predict_cluster" \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "gender": "Male", "income": 65000, "spending_score": 75, "region": "North", "purchase_frequency": 24, "avg_order_value": 450, "recency": 15}'
```

---

## 🐛 Troubleshooting

### Streamlit won't start
```bash
# Check if port 8501 is in use
lsof -i :8501

# Kill if needed
kill -9 <PID>

# Try again
cd /app/customer_segmentation && ./run_streamlit.sh
```

### API not responding
```bash
# Check backend status
sudo supervisorctl status backend

# Restart if needed
sudo supervisorctl restart backend

# Check logs
tail -50 /var/log/supervisor/backend.err.log
```

### React frontend issues
```bash
# Check frontend status
sudo supervisorctl status frontend

# Restart
sudo supervisorctl restart frontend
```

### Model not found
```bash
# Retrain
cd /app/customer_segmentation
python notebooks/EDA_and_Training.py
```

---

## 📊 Understanding Results

### Cluster 0 (Budget-Conscious)
- Lower income (~$47K)
- Lower spending score (~40)
- Fewer purchases (~8/year)
- **Action**: Discounts, promotions, value offers

### Cluster 1 (Premium)
- Higher income (~$78K)
- Higher spending score (~73)
- More purchases (~15/year)
- **Action**: Exclusive offers, VIP programs

---

## 🎉 You're All Set!

Choose your preferred interface and start predicting customer segments!

**Recommended**: Start with the React frontend (already running) for quick demo, then explore Streamlit for deeper analytics.

For questions, see: `/app/customer_segmentation/README.md`
