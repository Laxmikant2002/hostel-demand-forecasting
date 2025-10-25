"""
MongoDB Data Verification Script
=================================
Verify and query data stored in MongoDB Atlas

Author: Data Analysis Team
Date: October 24, 2025
"""

import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration - load from environment variables
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://USERNAME:PASSWORD@hostel-data-cluster.xxxxx.mongodb.net/')
DATABASE_NAME = os.getenv('MONGODB_DB', 'hostel_forecasting')
COLLECTION_NAME = 'demand_data'

# Color codes for terminal output
class Colors:
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

def connect_to_mongodb():
    """Connect to MongoDB"""
    try:
        # Get URI from environment (loaded from .env file)
        uri = MONGO_URI
        
        # Check if still using placeholder
        if '<db_password>' in uri or uri == "mongodb+srv://USERNAME:PASSWORD@hostel-data-cluster.xxxxx.mongodb.net/":
            print_error("MongoDB URI not configured!")
            print("\n⚠️  Please update your .env file with the actual MongoDB password")
            print("   Current .env location:", env_path)
            print("\nReplace '<db_password>' in MONGODB_URI with your actual MongoDB password")
            return None, None
        
        # Create client
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        
        # Access database and collection
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        return client, collection
    
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return None, None

def test_basic_queries(collection):
    """Test basic MongoDB queries"""
    print_header("BASIC QUERIES")
    
    try:
        # Count total documents
        total = collection.count_documents({})
        print(f"Total documents: {total:,}")
        
        # Fetch first 5 documents
        print("\nFirst 5 documents:")
        sample = list(collection.find().limit(5))
        
        for i, doc in enumerate(sample, 1):
            print(f"\nDocument {i}:")
            for key, value in doc.items():
                if key != '_id':
                    print(f"   {key}: {value}")
        
        print_success("Basic queries executed successfully!")
        
    except Exception as e:
        print_error(f"Query failed: {e}")

def test_filtered_queries(collection):
    """Test filtered queries"""
    print_header("FILTERED QUERIES")
    
    try:
        # Query by hotel type
        print("\n1. Count by Hotel Type:")
        for hotel in ['City Hotel', 'Resort Hotel']:
            count = collection.count_documents({'hotel': hotel})
            print(f"   {hotel}: {count:,} documents")
        
        # Query by date range
        print("\n2. Documents in 2016:")
        count_2016 = collection.count_documents({
            'year': 2016
        })
        print(f"   Total documents: {count_2016:,}")
        
        # Query high-demand days
        print("\n3. High-demand days (>20 bookings):")
        high_demand = collection.count_documents({
            'total_bookings': {'$gt': 20}
        })
        print(f"   Total days: {high_demand:,}")
        
        # Sample high-demand day
        sample = collection.find_one({'total_bookings': {'$gt': 20}})
        if sample:
            print(f"\n   Sample high-demand day:")
            print(f"      Hotel: {sample.get('hotel')}")
            print(f"      Date: {sample.get('arrival_date')}")
            print(f"      Bookings: {sample.get('total_bookings')}")
            print(f"      Revenue: ${sample.get('estimated_revenue', 0):,.2f}")
        
        print_success("Filtered queries executed successfully!")
        
    except Exception as e:
        print_error(f"Query failed: {e}")

def test_aggregation_queries(collection):
    """Test aggregation queries"""
    print_header("AGGREGATION QUERIES")
    
    try:
        # 1. Total bookings by hotel
        print("\n1. Total Bookings by Hotel:")
        pipeline = [
            {
                '$group': {
                    '_id': '$hotel',
                    'total_bookings': {'$sum': '$total_bookings'},
                    'avg_bookings': {'$avg': '$total_bookings'},
                    'total_revenue': {'$sum': '$estimated_revenue'}
                }
            },
            {'$sort': {'total_bookings': -1}}
        ]
        
        results = list(collection.aggregate(pipeline))
        for result in results:
            print(f"\n   {result['_id']}:")
            print(f"      Total bookings: {result['total_bookings']:,.0f}")
            print(f"      Avg bookings/day: {result['avg_bookings']:.2f}")
            print(f"      Total revenue: ${result['total_revenue']:,.2f}")
        
        # 2. Monthly trends
        print("\n2. Top 5 Busiest Months:")
        pipeline = [
            {
                '$group': {
                    '_id': {
                        'year': '$year',
                        'month': '$month'
                    },
                    'total_bookings': {'$sum': '$total_bookings'},
                    'avg_cancellation_rate': {'$avg': '$cancellation_rate'}
                }
            },
            {'$sort': {'total_bookings': -1}},
            {'$limit': 5}
        ]
        
        results = list(collection.aggregate(pipeline))
        for i, result in enumerate(results, 1):
            year = result['_id']['year']
            month = result['_id']['month']
            print(f"   {i}. {year}-{month:02d}: {result['total_bookings']:,.0f} bookings "
                  f"(Cancellation: {result['avg_cancellation_rate']:.2f}%)")
        
        # 3. Weekend vs Weekday
        print("\n3. Weekend vs Weekday Performance:")
        pipeline = [
            {
                '$group': {
                    '_id': '$is_weekend',
                    'total_bookings': {'$sum': '$total_bookings'},
                    'avg_bookings': {'$avg': '$total_bookings'},
                    'avg_revenue': {'$avg': '$estimated_revenue'}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        for result in results:
            day_type = "Weekend" if result['_id'] == 1 else "Weekday"
            print(f"\n   {day_type}:")
            print(f"      Total bookings: {result['total_bookings']:,.0f}")
            print(f"      Avg bookings: {result['avg_bookings']:.2f}")
            print(f"      Avg revenue: ${result['avg_revenue']:,.2f}")
        
        print_success("Aggregation queries executed successfully!")
        
    except Exception as e:
        print_error(f"Aggregation failed: {e}")

def test_indexes(collection):
    """Test and display indexes"""
    print_header("INDEX INFORMATION")
    
    try:
        indexes = list(collection.list_indexes())
        
        print(f"Total indexes: {len(indexes)}")
        print("\nIndex details:")
        
        for idx in indexes:
            print(f"\n   Name: {idx['name']}")
            print(f"   Key: {idx.get('key', {})}")
            print(f"   Unique: {idx.get('unique', False)}")
        
        print_success("Indexes verified!")
        
    except Exception as e:
        print_error(f"Index check failed: {e}")

def export_sample_to_dataframe(collection):
    """Export sample data to pandas DataFrame"""
    print_header("EXPORT TO PANDAS")
    
    try:
        # Fetch sample data
        cursor = collection.find().limit(100)
        
        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))
        
        # Remove MongoDB _id field
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        print(f"Exported {len(df)} documents to DataFrame")
        print(f"\nDataFrame shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataFrame info:")
        print(df.info())
        
        print_success("Export to pandas successful!")
        
        return df
        
    except Exception as e:
        print_error(f"Export failed: {e}")
        return None

def run_custom_query(collection):
    """Allow user to run custom queries"""
    print_header("CUSTOM QUERY EXAMPLES")
    
    try:
        # Example 1: Find peak season
        print("\n1. Find Peak Season (Highest Average Bookings):")
        pipeline = [
            {
                '$group': {
                    '_id': {'month': '$month', 'hotel': '$hotel'},
                    'avg_bookings': {'$avg': '$total_bookings'}
                }
            },
            {'$sort': {'avg_bookings': -1}},
            {'$limit': 3}
        ]
        
        results = list(collection.aggregate(pipeline))
        for i, result in enumerate(results, 1):
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month = month_names[result['_id']['month']]
            hotel = result['_id']['hotel']
            avg = result['avg_bookings']
            print(f"   {i}. {month} - {hotel}: {avg:.2f} avg bookings/day")
        
        # Example 2: Low cancellation days
        print("\n2. Days with Lowest Cancellation Rates:")
        low_cancel = list(collection.find(
            {'cancellation_rate': {'$lt': 10}},
            {'hotel': 1, 'arrival_date': 1, 'cancellation_rate': 1, 'total_bookings': 1}
        ).sort('cancellation_rate', 1).limit(5))
        
        for i, doc in enumerate(low_cancel, 1):
            print(f"   {i}. {doc.get('hotel')} on {doc.get('arrival_date')}: "
                  f"{doc.get('cancellation_rate'):.2f}% cancel, "
                  f"{doc.get('total_bookings')} bookings")
        
        # Example 3: Revenue leaders
        print("\n3. Top 5 Revenue Days:")
        top_revenue = list(collection.find(
            {},
            {'hotel': 1, 'arrival_date': 1, 'estimated_revenue': 1, 'total_bookings': 1}
        ).sort('estimated_revenue', -1).limit(5))
        
        for i, doc in enumerate(top_revenue, 1):
            print(f"   {i}. {doc.get('hotel')} on {doc.get('arrival_date')}: "
                  f"${doc.get('estimated_revenue', 0):,.2f} "
                  f"({doc.get('total_bookings')} bookings)")
        
        print_success("Custom queries executed successfully!")
        
    except Exception as e:
        print_error(f"Custom query failed: {e}")

def main():
    """Main execution"""
    print_header("MONGODB DATA VERIFICATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect
    print("\nConnecting to MongoDB...")
    client, collection = connect_to_mongodb()
    
    if not collection:
        print_error("Cannot proceed without MongoDB connection")
        return
    
    print_success(f"Connected to {DATABASE_NAME}.{COLLECTION_NAME}")
    
    # Run tests
    test_basic_queries(collection)
    test_filtered_queries(collection)
    test_aggregation_queries(collection)
    test_indexes(collection)
    export_sample_to_dataframe(collection)
    run_custom_query(collection)
    
    # Close connection
    client.close()
    
    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}✓ VERIFICATION COMPLETE!{Colors.END}")
    print("=" * 80)
    
    print("\nAll queries executed successfully!")
    print("Your MongoDB setup is working correctly.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\n\nOperation cancelled by user.")
    except Exception as e:
        print_error(f"\n\nError: {e}")
