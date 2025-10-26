"""
Hotel Demand Forecasting Models
==============================
Build and train time series forecasting models for hotel demand prediction.

Models implemented:
1. Prophet (Facebook) - For overall trend and seasonality
2. SARIMA - For seasonal patterns
3. XGBoost - For feature-based prediction

Author: Data Analysis Team
Date: October 25, 2025
"""

import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pathlib import Path
import joblib

# Machine Learning & Time Series
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# MongoDB Configuration
MONGO_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('MONGODB_DB', 'hostel_forecasting')
COLLECTION_NAME = 'demand_data'

# Model save paths
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

class Colors:
    """Terminal color codes for output formatting"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print("=" * 80)

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def load_data_from_mongodb():
    """Load data from MongoDB and prepare for modeling"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Fetch all data
        cursor = collection.find()
        
        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))
        
        # Close connection
        client.close()
        
        # Prepare data for modeling
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        df = df.sort_values('arrival_date')
        
        print_success(f"Loaded {len(df):,} records from MongoDB")
        return df
    
    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return None

def prepare_daily_data(df):
    """Aggregate data to daily level for time series modeling"""
    # Calculate total guests per booking
    df['actual_guests'] = df['adults'] + df.get('children', 0).fillna(0) + df.get('babies', 0).fillna(0)
    
    daily_data = df.groupby(['arrival_date', 'hotel']).agg({
        'actual_guests': 'sum',
        'is_canceled': 'sum',
        'stays_in_weekend_nights': 'sum',
        'stays_in_week_nights': 'sum',
        'adr': 'mean'
    }).reset_index()
    
    # Calculate total nights and revenue
    daily_data['total_nights'] = daily_data['stays_in_weekend_nights'] + daily_data['stays_in_week_nights']
    daily_data['revenue'] = daily_data['adr'] * daily_data['actual_guests'] * daily_data['total_nights']
    
    # Split by hotel type
    city_hotel = daily_data[daily_data['hotel'] == 'City Hotel']
    resort_hotel = daily_data[daily_data['hotel'] == 'Resort Hotel']
    
    return city_hotel, resort_hotel

def train_prophet_model(df, hotel_type):
    """Train Facebook Prophet model for time series forecasting"""
    try:
        # Prepare data for Prophet
        prophet_df = df[['arrival_date', 'actual_guests']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(prophet_df)
        
        # Save model
        model_path = MODELS_DIR / f'prophet_{hotel_type.lower().replace(" ", "_")}.joblib'
        joblib.dump(model, model_path)
        
        print_success(f"Trained and saved Prophet model for {hotel_type}")
        return model
        
    except Exception as e:
        print_error(f"Failed to train Prophet model for {hotel_type}: {e}")
        return None

def train_sarima_model(df, hotel_type):
    """Train SARIMA model for time series forecasting"""
    try:
        # Prepare data
        y = df.set_index('arrival_date')['actual_guests']
        
        # Initialize and train model
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False
        )
        results = model.fit()
        
        # Save model
        model_path = MODELS_DIR / f'sarima_{hotel_type.lower().replace(" ", "_")}.joblib'
        joblib.dump(results, model_path)
        
        print_success(f"Trained and saved SARIMA model for {hotel_type}")
        return results
        
    except Exception as e:
        print_error(f"Failed to train SARIMA model for {hotel_type}: {e}")
        return None

def prepare_features(df, last_known_values=None):
    """Prepare features for XGBoost model"""
    features_df = pd.DataFrame()
    
    # Extract time-based features
    features_df['year'] = df['arrival_date'].dt.year
    features_df['month'] = df['arrival_date'].dt.month
    features_df['day_of_week'] = df['arrival_date'].dt.dayofweek
    features_df['day_of_year'] = df['arrival_date'].dt.dayofyear
    features_df['is_weekend'] = df['arrival_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    if last_known_values is not None:
        # For future predictions, use the last known values
        features_df['rolling_mean_7d'] = last_known_values['rolling_mean_7d']
        features_df['rolling_std_7d'] = last_known_values['rolling_std_7d']
    else:
        # For training data, calculate rolling statistics
        features_df['rolling_mean_7d'] = df['actual_guests'].rolling(7, min_periods=1).mean()
        features_df['rolling_std_7d'] = df['actual_guests'].rolling(7, min_periods=1).std()
    
    # Fill missing values from rolling calculations
    features_df = features_df.fillna(features_df.mean())
    
    return features_df

def train_xgboost_model(df, hotel_type):
    """Train XGBoost model for demand prediction"""
    try:
        # Prepare features
        features_df = prepare_features(df)
        
        # Define features and target
        feature_cols = [
            'year', 'month', 'day_of_week', 'day_of_year',
            'rolling_mean_7d', 'rolling_std_7d', 'is_weekend'
        ]
        
        X = features_df[feature_cols]
        y = df['actual_guests']        # Initialize and train model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1
        )
        model.fit(X, y)
        
        # Save model
        model_path = MODELS_DIR / f'xgboost_{hotel_type.lower().replace(" ", "_")}.joblib'
        joblib.dump(model, model_path)
        
        print_success(f"Trained and saved XGBoost model for {hotel_type}")
        return model, feature_cols
        
    except Exception as e:
        print_error(f"Failed to train XGBoost model for {hotel_type}: {e}")
        return None, None

def evaluate_models(df, prophet_model, sarima_model, xgb_model, feature_cols, hotel_type):
    """Evaluate all models and compare performance"""
    try:
        # Prepare test data (last 30 days)
        test_size = 30
        train_df = df[:-test_size]
        test_df = df[-test_size:]
        
        results = {
            'prophet': {'predictions': [], 'metrics': {}},
            'sarima': {'predictions': [], 'metrics': {}},
            'xgboost': {'predictions': [], 'metrics': {}}
        }
        
        # Prophet predictions
        prophet_test = pd.DataFrame({'ds': test_df['arrival_date']})
        prophet_pred = prophet_model.predict(prophet_test)
        results['prophet']['predictions'] = prophet_pred['yhat'].values
        
        # SARIMA predictions
        sarima_pred = sarima_model.get_forecast(test_size)
        results['sarima']['predictions'] = sarima_pred.predicted_mean.values
        
        # XGBoost predictions
        xgb_test = prepare_features(test_df)
        results['xgboost']['predictions'] = xgb_model.predict(xgb_test[feature_cols])
        
        # Calculate metrics
        actual = test_df['actual_guests'].values
        
        for model_name in results:
            pred = results[model_name]['predictions']
            # Ensure predictions have same length as actual
            pred = pred[:len(actual)]
            results[model_name]['metrics'] = {
                'MAE': mean_absolute_error(actual, pred),
                'RMSE': np.sqrt(mean_squared_error(actual, pred)),
                'R2': r2_score(actual, pred)
            }
        
        # Save evaluation results
        eval_path = MODELS_DIR / f'evaluation_{hotel_type.lower().replace(" ", "_")}.txt'
        with open(eval_path, 'w') as f:
            f.write(f"Model Evaluation Results for {hotel_type}\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name.upper()} Model:\n")
                for metric, value in result['metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        print_success(f"Saved evaluation results for {hotel_type}")
        return results
        
    except Exception as e:
        print_error(f"Failed to evaluate models for {hotel_type}: {e}")
        return None

def save_forecasts_direct(hotel_type, days_to_forecast=90):
    """Generate forecasts directly and save them"""
    try:
        # Load models and data
        prophet_file = MODELS_DIR / f'prophet_{hotel_type.lower().replace(" ", "_")}.joblib'
        sarima_file = MODELS_DIR / f'sarima_{hotel_type.lower().replace(" ", "_")}.joblib'
        xgb_file = MODELS_DIR / f'xgboost_{hotel_type.lower().replace(" ", "_")}.joblib'
        
        prophet_model = joblib.load(prophet_file)
        sarima_model = joblib.load(sarima_file)
        xgb_model = joblib.load(xgb_file)
        
        # Load data for last known values
        df = load_data_from_mongodb()
        if df is None:
            raise Exception("Failed to load historical data")
            
        # Filter for hotel type and prepare data
        hotel_data = df[df['hotel'] == hotel_type]
        last_known_values = {
            'rolling_mean_7d': hotel_data['actual_guests'].iloc[-7:].mean(),
            'rolling_std_7d': hotel_data['actual_guests'].iloc[-7:].std()
        }
        
        # Generate future dates
        last_date = hotel_data['arrival_date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_to_forecast,
            freq='D'
        )
        
        # Prophet forecast
        prophet_future = pd.DataFrame({'ds': future_dates})
        prophet_forecast = models['prophet'].predict(prophet_future)
        forecasts['prophet'] = prophet_forecast['yhat'].values
        
        # SARIMA forecast
        sarima_forecast = models['sarima'].get_forecast(days_to_forecast)
        forecasts['sarima'] = sarima_forecast.predicted_mean.values
        
        # XGBoost forecast
        future_df = pd.DataFrame({'arrival_date': future_dates})
        future_features = prepare_features(future_df, last_known_values=last_known_values)
        xgb_forecast = xgb_model.predict(future_features)
        
        # Save forecasts
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'prophet_forecast': forecasts['prophet'],
            'sarima_forecast': forecasts['sarima'],
            'xgboost_forecast': forecasts['xgboost']
        })
        
        forecast_path = MODELS_DIR / f'forecasts_{hotel_type.lower().replace(" ", "_")}.csv'
        forecast_df.to_csv(forecast_path, index=False)
        
        print_success(f"Generated and saved forecasts for {hotel_type}")
        return forecasts
        
    except Exception as e:
        print_error(f"Failed to generate forecasts for {hotel_type}: {e}")
        return None

def get_last_known_values(df):
    """Get last known values for rolling statistics"""
    return {
        'rolling_mean_7d': df['actual_guests'].iloc[-7:].mean(),
        'rolling_std_7d': df['actual_guests'].iloc[-7:].std()
    }

def save_forecasts_direct(hotel_type, days_to_forecast=90):
    """Generate and save forecasts directly from saved models"""
    try:
        # Load models and data
        prophet_file = MODELS_DIR / f'prophet_{hotel_type.lower().replace(" ", "_")}.joblib'
        sarima_file = MODELS_DIR / f'sarima_{hotel_type.lower().replace(" ", "_")}.joblib'
        xgb_file = MODELS_DIR / f'xgboost_{hotel_type.lower().replace(" ", "_")}.joblib'
        
        prophet_model = joblib.load(prophet_file)
        sarima_model = joblib.load(sarima_file)
        xgb_model = joblib.load(xgb_file)
        
        # Load historical data for last known values
        df = load_data_from_mongodb()
        if df is None:
            raise Exception("Failed to load historical data")
            
        # Filter for hotel type
        df = df[df['hotel'] == hotel_type]
        last_known_values = get_last_known_values(df)
        
        # Generate future dates
        last_date = datetime.now()
        future_dates = pd.date_range(
            start=last_date,
            periods=days_to_forecast,
            freq='D'
        )
        
        # Prophet forecast
        prophet_future = pd.DataFrame({'ds': future_dates})
        prophet_forecast = prophet_model.predict(prophet_future)['yhat']
        
        # SARIMA forecast
        sarima_forecast = sarima_model.get_forecast(steps=days_to_forecast).predicted_mean
        
        # XGBoost forecast (prepare features)
        future_df = pd.DataFrame({'arrival_date': future_dates})
        future_features = prepare_features(future_df)
        xgb_forecast = xgb_model.predict(future_features)
        
        # Create and save forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'prophet_forecast': prophet_forecast,
            'sarima_forecast': sarima_forecast,
            'xgboost_forecast': xgb_forecast
        })
        
        forecast_path = MODELS_DIR / f'forecasts_{hotel_type.lower().replace(" ", "_")}.csv'
        forecast_df.to_csv(forecast_path, index=False)
        
        print_success(f"Generated and saved forecasts for {hotel_type}")
        return True
        
    except Exception as e:
        print_error(f"Failed to generate forecasts for {hotel_type}: {str(e)}")
        return False

def plot_forecasts(df, hotel_type):
    """Create visualization of historical data and forecasts"""
    try:
        # Load forecasts
        forecast_path = MODELS_DIR / f'forecasts_{hotel_type.lower().replace(" ", "_")}.csv'
        forecasts_df = pd.read_csv(forecast_path)
        forecasts_df['date'] = pd.to_datetime(forecasts_df['date'])
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['arrival_date'],
                y=df['actual_guests'],
                name="Historical",
                line=dict(color='black')
            )
        )
        
        # Add forecasts
        colors = ['blue', 'red', 'green']
        forecast_names = ['prophet', 'sarima', 'xgboost']
        
        for name, color in zip(forecast_names, colors):
            fig.add_trace(
                go.Scatter(
                    x=forecasts_df['date'],
                    y=forecasts_df[f'{name}_forecast'],
                    name=f"{name.capitalize()} Forecast",
                    line=dict(color=color, dash='dash')
                )
            )
        
        # Add forecasts
        colors = ['blue', 'red', 'green']
        forecast_names = ['prophet', 'sarima', 'xgboost']
        
        for name, color in zip(forecast_names, colors):
            fig.add_trace(
                go.Scatter(
                    x=forecasts_df['date'],
                    y=forecasts_df[f'{name}_forecast'],
                    name=f"{name.capitalize()} Forecast",
                    line=dict(color=color, dash='dash')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{hotel_type} - Demand Forecasts",
            xaxis_title="Date",
            yaxis_title="Total Guests",
            hovermode='x unified'
        )
        
        # Save plot
        plot_path = MODELS_DIR / f'forecast_plot_{hotel_type.lower().replace(" ", "_")}.html'
        fig.write_html(str(plot_path))
        
        print_success(f"Created and saved forecast plot for {hotel_type}")
        
    except Exception as e:
        print_error(f"Failed to create forecast plot for {hotel_type}: {e}")

def main():
    """Main execution function"""
    print_header("HOTEL DEMAND FORECASTING")
    start_time = datetime.now()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print_header("Step 1: Loading Data")
    df = load_data_from_mongodb()
    if df is None:
        return
    
    # Prepare daily data
    print_header("Step 2: Preparing Data")
    city_hotel, resort_hotel = prepare_daily_data(df)
    
    for hotel_data, hotel_type in [(city_hotel, "City Hotel"), (resort_hotel, "Resort Hotel")]:
        print_header(f"Training Models for {hotel_type}")
        
        # Train models
        prophet_model = train_prophet_model(hotel_data, hotel_type)
        sarima_model = train_sarima_model(hotel_data, hotel_type)
        xgb_model, feature_cols = train_xgboost_model(hotel_data, hotel_type)
        
        if not all([prophet_model, sarima_model, xgb_model]):
            print_error(f"Failed to train all models for {hotel_type}")
            continue
        
        # Evaluate models
        evaluation_results = evaluate_models(
            hotel_data,
            prophet_model,
            sarima_model,
            xgb_model,
            feature_cols,
            hotel_type
        )
        
        # Generate forecasts
        if save_forecasts_direct(hotel_type):
            # Create visualizations
            plot_forecasts(hotel_data, hotel_type)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("MODELING COMPLETE")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Results saved in: {MODELS_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\nOperation cancelled by user")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")