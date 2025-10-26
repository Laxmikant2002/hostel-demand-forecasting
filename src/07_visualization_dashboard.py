"""
Hotel Demand Forecasting Dashboard
================================
Create an interactive dashboard using Streamlit to visualize
forecasting results and insights.

Author: Data Analysis Team
Date: October 25, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib
from pathlib import Path

# Constants
MODELS_DIR = Path(__file__).parent.parent / 'models'

def load_model(model_type, hotel_type):
    """Load saved model from disk"""
    model_path = MODELS_DIR / f'{model_type}_{hotel_type.lower().replace(" ", "_")}.joblib'
    return joblib.load(model_path)

def load_forecasts(hotel_type):
    """Load saved forecasts from disk"""
    forecast_path = MODELS_DIR / f'forecasts_{hotel_type.lower().replace(" ", "_")}.csv'
    return pd.read_csv(forecast_path)

def load_evaluation_results(hotel_type):
    """Load model evaluation results"""
    eval_path = MODELS_DIR / f'evaluation_{hotel_type.lower().replace(" ", "_")}.txt'
    results = {}
    with open(eval_path, 'r') as f:
        content = f.read()
        # Parse the evaluation metrics from the text file
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':')
                try:
                    results[key.strip()] = float(value.strip())
                except:
                    continue
    return results

def plot_forecast_comparison(forecasts_df, hotel_type):
    """Create comparison plot of different model forecasts"""
    fig = go.Figure()

    # Add traces for each model's forecast
    fig.add_trace(
        go.Scatter(
            x=forecasts_df['date'],
            y=forecasts_df['prophet_forecast'],
            name='Prophet',
            line=dict(color='blue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecasts_df['date'],
            y=forecasts_df['sarima_forecast'],
            name='SARIMA',
            line=dict(color='red')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecasts_df['date'],
            y=forecasts_df['xgboost_forecast'],
            name='XGBoost',
            line=dict(color='green')
        )
    )

    # Update layout
    fig.update_layout(
        title=f'{hotel_type} - Demand Forecasts Comparison',
        xaxis_title='Date',
        yaxis_title='Predicted Number of Guests',
        hovermode='x unified'
    )
    
    return fig

def plot_model_performance(eval_results):
    """Create bar plot of model performance metrics"""
    models = ['Prophet', 'SARIMA', 'XGBoost']
    metrics = ['MAE', 'RMSE', 'R2']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=metrics
    )
    
    colors = ['blue', 'red', 'green']
    
    for i, metric in enumerate(metrics, 1):
        y_values = [eval_results[f'{model}_{metric}'] for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=y_values,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        height=400,
        title_text='Model Performance Comparison'
    )
    
    return fig

def main():
    st.set_page_config(
        page_title='Hotel Demand Forecasting',
        page_icon='📊',
        layout='wide'
    )
    
    st.title('Hotel Demand Forecasting Dashboard')
    st.markdown('''
    This dashboard presents the results of our demand forecasting models for both City Hotel and Resort Hotel.
    The forecasts are generated using three different models: Prophet, SARIMA, and XGBoost.
    ''')
    
    # Hotel Type Selection
    hotel_type = st.selectbox(
        'Select Hotel Type',
        ['City Hotel', 'Resort Hotel']
    )
    
    try:
        # Load forecasts and evaluation results
        forecasts_df = load_forecasts(hotel_type)
        eval_results = load_evaluation_results(hotel_type)
        
        # Forecast Comparison Plot
        st.subheader('Demand Forecasts')
        forecast_fig = plot_forecast_comparison(forecasts_df, hotel_type)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Model Performance Comparison
        st.subheader('Model Performance')
        performance_fig = plot_model_performance(eval_results)
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Detailed Metrics
        st.subheader('Detailed Metrics')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Prophet MAE', f"{eval_results['Prophet_MAE']:.2f}")
            st.metric('Prophet RMSE', f"{eval_results['Prophet_RMSE']:.2f}")
            st.metric('Prophet R²', f"{eval_results['Prophet_R2']:.3f}")
        
        with col2:
            st.metric('SARIMA MAE', f"{eval_results['SARIMA_MAE']:.2f}")
            st.metric('SARIMA RMSE', f"{eval_results['SARIMA_RMSE']:.2f}")
            st.metric('SARIMA R²', f"{eval_results['SARIMA_R2']:.3f}")
        
        with col3:
            st.metric('XGBoost MAE', f"{eval_results['XGBoost_MAE']:.2f}")
            st.metric('XGBoost RMSE', f"{eval_results['XGBoost_RMSE']:.2f}")
            st.metric('XGBoost R²', f"{eval_results['XGBoost_R2']:.3f}")
        
        # Forecast Data Table
        st.subheader('Forecast Data')
        st.dataframe(forecasts_df)
        
    except Exception as e:
        st.error(f'Error loading data: {str(e)}')
        st.error('Please ensure that the forecasting models have been run and saved correctly.')

if __name__ == '__main__':
    main()