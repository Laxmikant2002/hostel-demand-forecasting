"""
Initial Data Exploration - Hotel Booking Demand Dataset
========================================================
This script performs initial data exploration and quality checks on the hotel booking dataset.

Author: Data Analysis Team
Date: October 24, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Configure visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("HOTEL BOOKING DEMAND - INITIAL DATA EXPLORATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING DATASET")
print("="*80)

# Load hotel booking CSV and display first 5 rows
df = pd.read_csv('data/raw/hotel_bookings.csv')
print("\n✅ Dataset loaded successfully!")
print(f"\n📊 First 5 rows of the dataset:")
print(df.head())

# ============================================================================
# 2. DATA STRUCTURE OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("2. DATA STRUCTURE OVERVIEW")
print("="*80)

print(f"\n📈 Dataset Shape:")
print(f"   Total Rows: {df.shape[0]:,}")
print(f"   Total Columns: {df.shape[1]}")

print(f"\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# ============================================================================
# 3. DATA QUALITY CHECK
# ============================================================================
print("\n" + "="*80)
print("3. DATA QUALITY CHECK")
print("="*80)

# Check for missing values and data types
print("\n🔍 Dataset Information:")
print(df.info())

print("\n\n❌ Missing Values Analysis:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing Percentage': missing_percent.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
    print(f"\n⚠️  Total columns with missing values: {len(missing_df)}")
else:
    print("✅ No missing values found!")

# ============================================================================
# 4. BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print("4. SUMMARY STATISTICS")
print("="*80)

# Show summary statistics
print("\n📊 Numerical Features Summary:")
print(df.describe())

print("\n📊 Categorical Features Summary:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols[:5]:  # Show first 5 categorical columns
    print(f"\n   {col}:")
    print(f"   {df[col].value_counts().head()}")

# ============================================================================
# 5. KEY COLUMNS FOR FORECASTING
# ============================================================================
print("\n" + "="*80)
print("5. KEY COLUMNS FOR FORECASTING")
print("="*80)

key_columns = [
    'arrival_date_year',
    'arrival_date_month', 
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'hotel',
    'is_canceled',
    'lead_time',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'adr'
]

print("\n🎯 Key Forecasting Columns:")
for col in key_columns:
    if col in df.columns:
        print(f"   ✅ {col}")
        if df[col].dtype in ['int64', 'float64']:
            print(f"      Range: {df[col].min()} to {df[col].max()}")
        else:
            print(f"      Unique values: {df[col].nunique()}")

# ============================================================================
# 6. BUSINESS METRICS
# ============================================================================
print("\n" + "="*80)
print("6. BUSINESS METRICS")
print("="*80)

print(f"\n🏨 Hotel Types:")
print(df['hotel'].value_counts())
print(f"\n   Total bookings: {len(df):,}")

print(f"\n❌ Cancellation Analysis:")
cancellation_rate = df['is_canceled'].mean() * 100
print(f"   Cancellation Rate: {cancellation_rate:.2f}%")
print(f"   Canceled Bookings: {df['is_canceled'].sum():,}")
print(f"   Completed Bookings: {(len(df) - df['is_canceled'].sum()):,}")

print(f"\n💰 Average Daily Rate (ADR):")
print(f"   Mean ADR: ${df['adr'].mean():.2f}")
print(f"   Median ADR: ${df['adr'].median():.2f}")
print(f"   Min ADR: ${df['adr'].min():.2f}")
print(f"   Max ADR: ${df['adr'].max():.2f}")

print(f"\n📅 Booking Date Range:")
print(f"   Years: {df['arrival_date_year'].min()} - {df['arrival_date_year'].max()}")
print(f"   Months: {df['arrival_date_month'].unique()}")

print(f"\n👥 Guest Composition:")
total_adults = df['adults'].sum()
total_children = df['children'].sum()
total_babies = df['babies'].sum()
print(f"   Total Adults: {total_adults:,}")
print(f"   Total Children: {total_children:,.0f}")
print(f"   Total Babies: {total_babies:,.0f}")
print(f"   Average party size: {(df['adults'] + df['children'] + df['babies']).mean():.2f}")

print(f"\n⏰ Lead Time Analysis:")
print(f"   Average Lead Time: {df['lead_time'].mean():.0f} days")
print(f"   Median Lead Time: {df['lead_time'].median():.0f} days")
print(f"   Max Lead Time: {df['lead_time'].max()} days")

# ============================================================================
# 7. DATA QUALITY ISSUES
# ============================================================================
print("\n" + "="*80)
print("7. POTENTIAL DATA QUALITY ISSUES")
print("="*80)

issues = []

# Check for zero or negative ADR
zero_adr = (df['adr'] <= 0).sum()
if zero_adr > 0:
    issues.append(f"⚠️  {zero_adr:,} bookings have zero or negative ADR")

# Check for zero guests
zero_guests = ((df['adults'] + df['children'] + df['babies']) == 0).sum()
if zero_guests > 0:
    issues.append(f"⚠️  {zero_guests:,} bookings have zero guests")

# Check for unrealistic stay durations
total_nights = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
zero_nights = (total_nights == 0).sum()
if zero_nights > 0:
    issues.append(f"⚠️  {zero_nights:,} bookings have zero nights stay")

if issues:
    print("\n⚠️  Issues Found:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ No major data quality issues detected!")

# ============================================================================
# 8. SAVE SUMMARY FINDINGS
# ============================================================================
print("\n" + "="*80)
print("8. SUMMARY FINDINGS")
print("="*80)

findings = f"""
DATA EXPLORATION SUMMARY
========================

Dataset Overview:
- Total Records: {df.shape[0]:,}
- Total Features: {df.shape[1]}
- Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB
- Date Range: {df['arrival_date_year'].min()} - {df['arrival_date_year'].max()}

Data Quality:
- Missing Values: {len(missing_df)} columns with missing data
- Most Missing: {missing_df.iloc[0]['Column'] if len(missing_df) > 0 else 'None'} ({missing_df.iloc[0]['Missing Percentage']:.2f}% if len(missing_df) > 0 else 0)

Key Metrics:
- Hotels: City Hotel ({df[df['hotel']=='City Hotel'].shape[0]:,}) | Resort Hotel ({df[df['hotel']=='Resort Hotel'].shape[0]:,})
- Cancellation Rate: {cancellation_rate:.2f}%
- Average ADR: ${df['adr'].mean():.2f}
- Average Lead Time: {df['lead_time'].mean():.0f} days
- Average Party Size: {(df['adults'] + df['children'] + df['babies']).mean():.2f} guests

Key Forecasting Features:
✅ arrival_date_year - Years: {df['arrival_date_year'].unique().tolist()}
✅ arrival_date_month - All 12 months present
✅ arrival_date_week_number - Week numbers: 1-53
✅ arrival_date_day_of_month - Days: 1-31
✅ hotel - 2 types: City Hotel, Resort Hotel
✅ is_canceled - Binary: 0 (not canceled), 1 (canceled)
✅ lead_time - Range: {df['lead_time'].min()}-{df['lead_time'].max()} days
✅ adr - Average Daily Rate (price metric)

Data Quality Issues:
{chr(10).join(['- ' + issue for issue in issues]) if issues else '✅ No major issues found'}

Recommendations for Next Steps:
1. Handle missing values in 'company' column (94% missing - consider dropping)
2. Investigate zero/negative ADR values
3. Create time-based features (date, day of week, season)
4. Analyze seasonal patterns and trends
5. Build forecasting models (Prophet, ARIMA, ML models)
"""

print(findings)

# Save findings to file
with open('data_exploration_findings.txt', 'w') as f:
    f.write(findings)

print("\n✅ Findings saved to 'data_exploration_findings.txt'")

print("\n" + "="*80)
print("✨ DATA EXPLORATION COMPLETE!")
print("="*80)
print("\n🎯 Next Steps:")
print("   1. Review the findings above")
print("   2. Plan data cleaning strategy")
print("   3. Create visualizations (run: 02_data_visualization.py)")
print("   4. Build forecasting models")
print("="*80)
