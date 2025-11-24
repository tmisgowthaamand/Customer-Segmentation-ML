import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_columns = None
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        """
        df_copy = df.copy()
        
        # Numerical columns - impute with median
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_copy[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df_copy[col] = self.imputers[col].fit_transform(df_copy[[col]]).ravel()
                else:
                    df_copy[col] = self.imputers[col].transform(df_copy[[col]]).ravel()
        
        # Categorical columns - impute with mode
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_copy[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                    df_copy[col] = self.imputers[col].fit_transform(df_copy[[col]]).ravel()
                else:
                    df_copy[col] = self.imputers[col].transform(df_copy[[col]]).ravel()
        
        return df_copy
    
    def detect_outliers(self, df, columns, threshold=3):
        """
        Detect outliers using Z-score method
        """
        outliers_dict = {}
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_dict[col] = (z_scores > threshold).sum()
        return outliers_dict
    
    def remove_outliers(self, df, columns, threshold=3):
        """
        Remove outliers using Z-score method
        """
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in [np.float64, np.int64]:
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy = df_copy[z_scores <= threshold]
        return df_copy
    
    def encode_categorical(self, df, categorical_cols):
        """
        Encode categorical variables
        """
        df_copy = df.copy()
        for col in categorical_cols:
            if col in df_copy.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
                else:
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
        return df_copy
    
    def scale_features(self, df, fit=True):
        """
        Scale features using StandardScaler
        """
        if fit:
            scaled_data = self.scaler.fit_transform(df)
        else:
            scaled_data = self.scaler.transform(df)
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def preprocess(self, df, remove_outliers=True, fit=True):
        """
        Complete preprocessing pipeline
        """
        # Save CustomerID if exists
        customer_ids = None
        if 'CustomerID' in df.columns:
            customer_ids = df['CustomerID'].copy()
            df = df.drop('CustomerID', axis=1)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Identify numerical and categorical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Detect and optionally remove outliers
        if remove_outliers and fit:
            outliers = self.detect_outliers(df_clean, numerical_cols)
            print(f"Outliers detected: {outliers}")
            df_clean = self.remove_outliers(df_clean, numerical_cols)
            print(f"Shape after removing outliers: {df_clean.shape}")
        
        # Encode categorical variables
        if categorical_cols:
            df_clean = self.encode_categorical(df_clean, categorical_cols)
        
        # Store feature columns
        if fit:
            self.feature_columns = df_clean.columns.tolist()
        
        # Scale features
        df_scaled = self.scale_features(df_clean, fit=fit)
        
        # Add CustomerID back if it existed
        if customer_ids is not None:
            df_scaled.insert(0, 'CustomerID', customer_ids.values)
        
        return df_scaled
