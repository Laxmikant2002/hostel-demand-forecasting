"""
Explore InsideAirbnb Data
==========================
Quick exploration of downloaded Airbnb data to understand structure and contents.

Author: Data Analysis Team
Date: October 24, 2025
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
AIRBNB_DATA_DIR = "data/raw/airbnb"

def explore_city_data(city_name):
    """Explore Airbnb data for a specific city"""
    
    print("\n" + "="*80)
    print(f"🏙️  EXPLORING: {city_name.upper()}")
    print("="*80)
    
    city_dir = os.path.join(AIRBNB_DATA_DIR, city_name)
    
    if not os.path.exists(city_dir):
        print(f"❌ No data found for {city_name}")
        return
    
    # Check which files exist
    files = {
        'listings': os.path.join(city_dir, 'listings.csv'),
        'calendar': os.path.join(city_dir, 'calendar.csv'),
        'reviews': os.path.join(city_dir, 'reviews.csv')
    }
    
    results = {}
    
    for dataset_name, filepath in files.items():
        if os.path.exists(filepath):
            print(f"\n📊 {dataset_name.upper()}.CSV")
            print("-" * 80)
            
            try:
                # Read the dataset
                df = pd.read_csv(filepath, nrows=3)  # Read first 3 rows for preview
                full_df = pd.read_csv(filepath)  # Read full for stats
                
                # Basic info
                print(f"✅ File exists: {filepath}")
                print(f"📏 Shape: {full_df.shape[0]:,} rows × {full_df.shape[1]} columns")
                print(f"💾 Memory: {full_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
                
                # Column names
                print(f"\n📋 Columns ({len(full_df.columns)}):")
                for i, col in enumerate(full_df.columns[:10], 1):  # Show first 10
                    dtype = full_df[col].dtype
                    null_count = full_df[col].isnull().sum()
                    null_pct = (null_count / len(full_df)) * 100
                    print(f"   {i:2d}. {col:30s} {str(dtype):10s} (Missing: {null_pct:.1f}%)")
                
                if len(full_df.columns) > 10:
                    print(f"   ... and {len(full_df.columns) - 10} more columns")
                
                # Sample data
                print(f"\n👀 First 3 Rows:")
                print(df.head(3).to_string())
                
                # Save for summary
                results[dataset_name] = {
                    'rows': full_df.shape[0],
                    'columns': full_df.shape[1],
                    'memory_mb': full_df.memory_usage(deep=True).sum() / (1024**2)
                }
                
            except Exception as e:
                print(f"❌ Error reading {dataset_name}.csv: {e}")
        else:
            print(f"\n❌ {dataset_name.upper()}.CSV - File not found")
    
    return results

def explore_all_cities():
    """Explore all downloaded Airbnb data"""
    
    print("\n" + "="*80)
    print("🌍 INSIDEAIRBNB DATA EXPLORATION")
    print("="*80)
    
    if not os.path.exists(AIRBNB_DATA_DIR):
        print(f"\n❌ Airbnb data directory not found: {AIRBNB_DATA_DIR}")
        print("   Run: python 02_download_airbnb.py first")
        return
    
    # Get list of cities
    cities = [d for d in os.listdir(AIRBNB_DATA_DIR) 
              if os.path.isdir(os.path.join(AIRBNB_DATA_DIR, d))]
    
    if not cities:
        print("\n❌ No city data found")
        return
    
    print(f"\nFound data for {len(cities)} cities: {', '.join(cities)}")
    
    # Explore each city
    all_results = {}
    for city in cities:
        results = explore_city_data(city)
        if results:
            all_results[city] = results
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    print(f"\n🏙️  Cities with Data:")
    for city, datasets in all_results.items():
        print(f"\n   {city.title()}:")
        for dataset_name, info in datasets.items():
            print(f"      - {dataset_name}: {info['rows']:,} rows, {info['columns']} cols, {info['memory_mb']:.2f} MB")
    
    print("\n" + "="*80)
    print("✨ EXPLORATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    explore_all_cities()
