"""
Dashboard Utility Functions
=========================
Helper functions for the hostel demand forecasting dashboard.

Important Date Column Conventions:
- Use 'ds' as standard date column name (Prophet requirement)
- All timestamps should be in datetime format
- MongoDB data uses 'ds' column
- All visualizations reference 'ds' for dates

Author: Data Analysis Team
Date: October 26, 2025
"""

def standardize_date_columns(df, date_column=None):
    """
    Standardize date columns in DataFrame to use 'ds' as column name
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Name of current date column (if not 'ds' or 'arrival_date')
    
    Returns:
        pd.DataFrame: DataFrame with standardized date column
    """
    df = df.copy()
    
    # Identify date column
    if date_column:
        old_col = date_column
    elif 'arrival_date' in df.columns:
        old_col = 'arrival_date'
    elif 'ds' in df.columns:
        return df  # Already standardized
    else:
        raise ValueError("No date column found or specified")
    
    # Convert and rename
    df['ds'] = pd.to_datetime(df[old_col])
    if old_col != 'ds':
        df = df.drop(old_col, axis=1)
    
    return df

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from pathlib import Path
from dotenv import load_dotenv
import joblib

# Load environment variables
load_dotenv()

def load_mongodb_data(start_date=None, end_date=None, hotel_type=None):
    """
    Load data from MongoDB with optional filters
    
    Args:
        start_date (datetime): Start date for filtering
        end_date (datetime): End date for filtering
        hotel_type (str): Hotel type filter
    
    Returns:
        pd.DataFrame: Filtered hotel booking data
    """
    try:
        # MongoDB connection (using environment variables)
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('MONGODB_DB')]
        collection = db[os.getenv('MONGODB_COLLECTION')]
        
        # Build query
        query = {}
        if start_date or end_date:
            query['ds'] = {}
            if start_date:
                query['ds']['$gte'] = start_date
            if end_date:
                query['ds']['$lte'] = end_date
        
        if hotel_type and hotel_type != "All Hotels":
            query['hotel'] = hotel_type
        
        # Execute query
        data = list(collection.find(query))
        df = pd.DataFrame(data)
        
        # Convert ObjectId to string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading MongoDB data: {str(e)}")

def load_model_predictions(model_type='prophet', hotel_type=None):
    """
    Load model predictions from saved files
    
    Args:
        model_type (str): Type of model ('prophet', 'arima', etc.)
        hotel_type (str): Hotel type filter
    
    Returns:
        pd.DataFrame: Model predictions
    """
    try:
        models_dir = Path(__file__).parent.parent / 'models'
        
        if hotel_type and hotel_type != "All Hotels":
            model_path = models_dir / f"{model_type}_{hotel_type.lower().replace(' ', '_')}.joblib"
        else:
            model_path = models_dir / f"{model_type}_all.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        
        # Get predictions (implementation depends on model type)
        if model_type == 'prophet':
            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)
            return forecast
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    except Exception as e:
        raise Exception(f"Error loading model predictions: {str(e)}")

def calculate_metrics(actuals, predictions):
    """
    Calculate forecast accuracy metrics
    
    Args:
        actuals (np.array): Actual values
        predictions (np.array): Predicted values
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
        'rmse': np.sqrt(np.mean((actuals - predictions) ** 2)),
        'mae': np.mean(np.abs(actuals - predictions))
    }
    return metrics

def format_metrics(metrics):
    """
    Format metrics for display
    
    Args:
        metrics (dict): Dictionary of metrics
    
    Returns:
        dict: Formatted metrics
    """
    return {
        'MAPE': f"{metrics['mape']:.2f}%",
        'RMSE': f"{metrics['rmse']:.2f}",
        'MAE': f"{metrics['mae']:.2f}"
    }