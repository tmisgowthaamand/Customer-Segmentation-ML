import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_customer_data(n_customers=1000):
    """
    Generate synthetic customer data for segmentation
    """
    # Customer IDs
    customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    
    # Demographics
    ages = np.random.normal(40, 15, n_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04])
    
    # Income (correlated with age)
    base_income = np.random.normal(50000, 20000, n_customers)
    age_factor = (ages - 18) / 62  # Normalize age
    income = base_income + (age_factor * 30000) + np.random.normal(0, 5000, n_customers)
    income = np.clip(income, 20000, 150000).astype(int)
    
    # Spending Score (1-100, inversely correlated with some income groups)
    spending_score = np.zeros(n_customers)
    for i in range(n_customers):
        if income[i] < 40000:
            spending_score[i] = np.random.normal(35, 15)
        elif income[i] < 70000:
            spending_score[i] = np.random.normal(50, 20)
        else:
            spending_score[i] = np.random.normal(75, 15)
    spending_score = np.clip(spending_score, 1, 100).astype(int)
    
    # Region
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_customers)
    
    # Purchase Frequency (per year)
    purchase_frequency = np.random.poisson(spending_score / 5, n_customers)
    purchase_frequency = np.clip(purchase_frequency, 1, 50)
    
    # Average Order Value
    aov = income / 100 + np.random.normal(0, 50, n_customers)
    aov = np.clip(aov, 50, 2000).astype(int)
    
    # Recency (days since last purchase)
    recency = np.random.exponential(30, n_customers).astype(int)
    recency = np.clip(recency, 1, 365)
    
    # Total Spend (derived feature)
    total_spend = (purchase_frequency * aov).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'Income': income,
        'SpendingScore': spending_score,
        'Region': regions,
        'PurchaseFrequency': purchase_frequency,
        'AvgOrderValue': aov,
        'Recency': recency,
        'TotalSpend': total_spend
    })
    
    # Add some missing values randomly (2-3%)
    for col in ['Age', 'Income', 'SpendingScore']:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_customer_data(1000)
    
    # Create data directory if it doesn't exist
    os.makedirs('/app/customer_segmentation/data', exist_ok=True)
    
    # Save to CSV
    output_path = '/app/customer_segmentation/data/customers.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData Info:")
    print(df.info())
