from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import uuid
from datetime import datetime, timezone


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignore MongoDB's _id field
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    # Exclude MongoDB's _id field from the query results
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks


# Customer Segmentation Models
class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str
    income: float = Field(..., ge=0)
    spending_score: int = Field(..., ge=1, le=100)
    region: str
    purchase_frequency: int = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    recency: int = Field(..., ge=0, le=365)

class ClusterPrediction(BaseModel):
    cluster: int
    cluster_size: int
    cluster_characteristics: dict


@api_router.post("/predict_cluster", response_model=ClusterPrediction)
async def predict_customer_cluster(customer: CustomerInput):
    """
    Predict which cluster a customer belongs to based on their attributes
    """
    try:
        import sys
        sys.path.append('/app/customer_segmentation')
        
        import pandas as pd
        import joblib
        from src.clustering_model import CustomerSegmentation
        
        # Load models
        model = CustomerSegmentation.load_model('/app/customer_segmentation/model/kmeans_model.pkl')
        preprocessor = joblib.load('/app/customer_segmentation/model/preprocessor.pkl')
        
        # Calculate total spend
        total_spend = customer.purchase_frequency * customer.avg_order_value
        
        # Create dataframe
        customer_data = pd.DataFrame({
            'Age': [customer.age],
            'Gender': [customer.gender],
            'Income': [customer.income],
            'SpendingScore': [customer.spending_score],
            'Region': [customer.region],
            'PurchaseFrequency': [customer.purchase_frequency],
            'AvgOrderValue': [customer.avg_order_value],
            'Recency': [customer.recency],
            'TotalSpend': [total_spend]
        })
        
        # Preprocess
        customer_processed = preprocessor.preprocess(customer_data, remove_outliers=False, fit=False)
        
        # Predict
        cluster = int(model.predict(customer_processed)[0])
        
        # Load reference data for cluster info
        df_ref = pd.read_csv('/app/customer_segmentation/data/customers_clustered.csv')
        cluster_data = df_ref[df_ref['Cluster'] == cluster]
        
        cluster_chars = {
            'avg_income': float(cluster_data['Income'].mean()),
            'avg_spending_score': float(cluster_data['SpendingScore'].mean()),
            'avg_total_spend': float(cluster_data['TotalSpend'].mean()),
            'avg_purchase_frequency': float(cluster_data['PurchaseFrequency'].mean()),
            'avg_recency': float(cluster_data['Recency'].mean())
        }
        
        return ClusterPrediction(
            cluster=cluster,
            cluster_size=len(cluster_data),
            cluster_characteristics=cluster_chars
        )
        
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()