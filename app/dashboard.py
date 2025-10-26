"""
Hostel Demand Forecasting Dashboard
==================================
Interactive dashboard for visualizing hotel booking demand forecasts.

Author: Data Analysis Team
Date: October 26, 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import json

# Add project root to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from app.utils import (
    load_mongodb_data,
    load_model_predictions,
    calculate_metrics,
    format_metrics
)

def create_demand_trend_plot(historical_data, predictions, hotel_type=None):
    """Create an interactive demand trend plot"""
    
    if hotel_type and hotel_type != "All Hotels":
        historical_data = historical_data[historical_data['hotel'] == hotel_type]
    
    # Prepare data for plotting
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['arrival_date'],
            y=historical_data['adr'],
            name='Historical Demand',
            line=dict(color='blue'),
            mode='lines'
        )
    )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=predictions['ds'],
            y=predictions['yhat'],
            name='Forecast',
            line=dict(color='red', dash='dash'),
            mode='lines'
        )
    )
    
    # Add prediction intervals
    fig.add_trace(
        go.Scatter(
            x=predictions['ds'].tolist() + predictions['ds'].tolist()[::-1],
            y=predictions['yhat_upper'].tolist() + predictions['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='95% Confidence Interval'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Demand Trend and Forecast {hotel_type if hotel_type else 'All Hotels'}",
        xaxis_title="Date",
        yaxis_title="Average Daily Rate (EUR)",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_performance_metrics(historical_data, predictions, hotel_type=None):
    """Create performance metrics visualization"""
    
    if hotel_type and hotel_type != "All Hotels":
        historical_data = historical_data[historical_data['hotel'] == hotel_type]
    
    # Calculate metrics
    metrics = calculate_metrics(
        historical_data['adr'].values,
        predictions['yhat'].values[:len(historical_data)]
    )
    
    # Create gauge charts for each metric
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("MAPE (%)", "RMSE", "MAE")
    )
    
    # Add MAPE gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics['mape'],
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "darkblue"}},
            domain={'row': 0, 'column': 0}
        ),
        row=1, col=1
    )
    
    # Add RMSE gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics['rmse'],
            gauge={'axis': {'range': [0, max(metrics['rmse'] * 2, 100)]},
                  'bar': {'color': "darkblue"}},
            domain={'row': 0, 'column': 1}
        ),
        row=1, col=2
    )
    
    # Add MAE gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics['mae'],
            gauge={'axis': {'range': [0, max(metrics['mae'] * 2, 50)]},
                  'bar': {'color': "darkblue"}},
            domain={'row': 0, 'column': 2}
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=f"Forecast Performance Metrics - {hotel_type if hotel_type else 'All Hotels'}",
        height=300
    )
    
    return fig

def create_seasonality_plot(historical_data, hotel_type=None):
    """Create a seasonal pattern visualization"""
    
    if hotel_type and hotel_type != "All Hotels":
        data = historical_data[historical_data['hotel'] == hotel_type].copy()
    else:
        data = historical_data.copy()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Monthly Seasonality", "Day of Week Pattern")
    )
    
    # Monthly pattern
    monthly_stats = data.groupby(data['arrival_date'].dt.month_name()).agg({
        'adr': 'mean'
    }).reset_index()
    
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    monthly_stats['month'] = pd.Categorical(monthly_stats['month'], categories=month_order, ordered=True)
    monthly_stats = monthly_stats.sort_values('month')
    
    fig.add_trace(
        go.Bar(
            x=monthly_stats['month'],
            y=monthly_stats['adr'],
            name='Monthly Average'
        ),
        row=1, col=1
    )
    
    # Daily pattern
    daily_stats = data.groupby(data['arrival_date'].dt.day_name()).agg({
        'adr': 'mean'
    }).reset_index()
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats['day'] = pd.Categorical(daily_stats['arrival_date'], categories=days_order, ordered=True)
    daily_stats = daily_stats.sort_values('day')
    
    fig.add_trace(
        go.Bar(
            x=days_order,
            y=daily_stats['adr'],
            name='Daily Average'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Seasonal Patterns - {hotel_type if hotel_type else 'All Hotels'}",
        showlegend=False,
        height=400
    )
    
    return fig

def prepare_download_data(historical_data, predictions, hotel_type=None):
    """Prepare data for download in various formats"""
    
    if hotel_type and hotel_type != "All Hotels":
        historical_data = historical_data[historical_data['hotel'] == hotel_type]
    
    # Merge historical and predicted data
    download_data = pd.merge(
        historical_data[['arrival_date', 'adr', 'hotel']],
        predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        left_on='arrival_date',
        right_on='ds',
        how='outer'
    )
    
    # Clean up columns
    download_data = download_data.rename(columns={
        'arrival_date': 'date',
        'adr': 'actual_demand',
        'yhat': 'predicted_demand',
        'yhat_lower': 'prediction_lower',
        'yhat_upper': 'prediction_upper'
    })
    
    # Drop redundant columns
    download_data = download_data.drop('ds', axis=1)
    
    return download_data

# Page configuration
st.set_page_config(
    page_title="Hostel Demand Forecasting",
    page_icon="🏨",
    layout="wide"
)

# Title and description
st.title("🏨 Hostel Demand Forecasting")
st.markdown("""
This dashboard provides interactive visualizations and insights for hotel booking demand.
Explore historical patterns, forecasts, and key metrics to optimize your revenue strategy.
""")

# Sidebar filters
st.sidebar.header("📊 Filters")

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime(2015, 7, 1), datetime(2017, 8, 31)),
    min_value=datetime(2015, 7, 1),
    max_value=datetime(2017, 8, 31)
)

# Hotel type selector
hotel_type = st.sidebar.selectbox(
    "Select Hotel Type",
    ["All Hotels", "City Hotel", "Resort Hotel"]
)

# View type selector
view_type = st.sidebar.radio(
    "Select View",
    ["Historical Analysis", "Forecast", "Seasonality", "Metrics"]
)

# Load data based on filters
try:
    # Load data (placeholder - will be replaced with MongoDB)
    df = pd.read_csv(ROOT_DIR / "data/processed/cleaned_hotel_demand.csv")
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    
    # Filter by date range and hotel type
    mask = (df['arrival_date'] >= pd.Timestamp(date_range[0])) & \
           (df['arrival_date'] <= pd.Timestamp(date_range[1]))
    
    if hotel_type != "All Hotels":
        mask &= (df['hotel'] == hotel_type)
    
    df_filtered = df[mask].copy()
    
    # Daily aggregation
    daily_bookings = df_filtered.groupby('arrival_date').agg({
        'hotel': 'count',
        'is_canceled': 'sum',
        'adr': 'mean'
    }).rename(columns={'hotel': 'total_bookings'})
    
    daily_bookings['cancellation_rate'] = daily_bookings['is_canceled'] / daily_bookings['total_bookings']
    
    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Bookings",
            f"{len(df_filtered):,}",
            f"{len(df_filtered) / len(df) * 100:.1f}% of all"
        )
    
    with col2:
        st.metric(
            "Average Daily Rate",
            f"${df_filtered['adr'].mean():.2f}",
            f"{(df_filtered['adr'].mean() / df['adr'].mean() - 1) * 100:+.1f}%"
        )
    
    with col3:
        st.metric(
            "Cancellation Rate",
            f"{df_filtered['is_canceled'].mean() * 100:.1f}%",
            f"{(df_filtered['is_canceled'].mean() - df['is_canceled'].mean()) * 100:+.1f}%"
        )
    
    with col4:
        st.metric(
            "Average Lead Time",
            f"{df_filtered['lead_time'].mean():.0f} days",
            f"{(df_filtered['lead_time'].mean() / df['lead_time'].mean() - 1) * 100:+.1f}%"
        )
    
    # Main visualization area
    if view_type == "Historical Analysis":
        st.header("📈 Historical Booking Patterns")
        
        # Daily bookings trend
        fig = px.line(
            daily_bookings,
            x=daily_bookings.index,
            y='total_bookings',
            title='Daily Booking Volume'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly patterns
        monthly_stats = df_filtered.groupby([
            df_filtered['arrival_date'].dt.year,
            df_filtered['arrival_date'].dt.month
        ]).agg({
            'hotel': 'count',
            'adr': 'mean',
            'is_canceled': 'mean'
        }).reset_index()
        
        monthly_stats.columns = ['year', 'month', 'bookings', 'adr', 'cancellation_rate']
        monthly_stats['date'] = pd.to_datetime(monthly_stats[['year', 'month']].assign(day=1))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                monthly_stats,
                x='date',
                y='bookings',
                title='Monthly Booking Volume'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                monthly_stats,
                x='date',
                y=['adr', 'cancellation_rate'],
                title='ADR and Cancellation Rate Trends'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Forecast":
        st.header("🔮 Demand Forecasts")
        
        # Load model predictions
        predictions = load_model_predictions(hotel_type=hotel_type)
        
        # Create demand trend plot
        fig_trend = create_demand_trend_plot(df_filtered, predictions, hotel_type)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Performance metrics
        fig_metrics = create_performance_metrics(df_filtered, predictions, hotel_type)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Add download button for forecast data
        forecast_data = prepare_download_data(df_filtered, predictions, hotel_type)
        st.download_button(
            label="📥 Download Forecast Data",
            data=forecast_data.to_csv(index=False),
            file_name=f"demand_forecast_{hotel_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    elif view_type == "Seasonality":
        st.header("🗓️ Seasonal Patterns")
        
        # Create seasonal patterns plot
        fig_season = create_seasonality_plot(df_filtered, hotel_type)
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Additional seasonal metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Peak Demand Days")
            peak_days = df_filtered.nlargest(5, 'adr')[['arrival_date', 'adr', 'hotel']]
            st.dataframe(peak_days)
        
        with col2:
            st.subheader("Seasonal Summary")
            seasonal_stats = df_filtered.groupby(
                df_filtered['arrival_date'].dt.quarter
            ).agg({
                'adr': ['mean', 'std', 'count']
            }).round(2)
            seasonal_stats.columns = ['Average Rate', 'Std Dev', 'Bookings']
            seasonal_stats.index = ['Q1', 'Q2', 'Q3', 'Q4']
            st.dataframe(seasonal_stats)
    
    else:  # Metrics view
        st.header("📊 Key Performance Metrics")
        
        # Calculate and display various metrics
        metrics = df_filtered.groupby('hotel').agg({
            'is_canceled': ['count', 'mean'],
            'adr': ['mean', 'std'],
            'lead_time': ['mean', 'median'],
            'total_nights': ['mean', 'sum']
        }).round(2)
        
        st.dataframe(metrics, use_container_width=True)
        
        # Allow downloading the filtered dataset
        st.download_button(
            label="📥 Download Filtered Data",
            data=df_filtered.to_csv(index=False),
            file_name=f"hotel_demand_{hotel_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error loading data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data refreshed daily • [View on GitHub](https://github.com/yourusername/hostel-demand-forecasting)")