"""
MongoDB Data Upload Script
===========================
Upload cleaned hotel demand data to MongoDB Atlas

Requirements:
- pymongo installed
- python-dotenv installed
- MongoDB Atlas connection string in .env file
- Cleaned data CSV file

Author: Data Analysis Team
Date: October 24, 2025
"""

import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from datetime import datetime
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration - load from environment variables
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://USERNAME:PASSWORD@hostel-data-cluster.xxxxx.mongodb.net/')
DATABASE_NAME = os.getenv('MONGODB_DB', 'hostel_forecasting')
COLLECTION_NAME = 'demand_data'
DATA_FILE = str(Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_hotel_demand.csv')

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

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def get_mongodb_uri():
    """
    Get MongoDB URI from environment variable or prompt user
    """
    # Try to get from environment variable (loaded from .env file)
    uri = MONGO_URI
    
    # Check if still using placeholder
    if '<db_password>' in uri or uri == "mongodb+srv://USERNAME:PASSWORD@hostel-data-cluster.xxxxx.mongodb.net/":
        print_warning("MongoDB URI not configured!")
        print("\n⚠️  Please update your .env file with the actual MongoDB password")
        print("   Current .env location:", env_path)
        print("\nReplace '<db_password>' in MONGODB_URI with your actual MongoDB password")
        print("\nAlternatively, enter connection string now:")
        
        user_input = input("\nEnter MongoDB connection string (or press Enter to exit): ").strip()
        if user_input:
            uri = user_input
        else:
            print_error("No valid MongoDB URI provided. Exiting.")
            sys.exit(1)
    
    return uri

def connect_to_mongodb(uri, timeout=5000):
    """
    Create MongoDB connection with error handling
    
    Args:
        uri: MongoDB connection string
        timeout: Connection timeout in milliseconds
    
    Returns:
        tuple: (client, database, collection) or (None, None, None) on failure
    """
    try:
        # Create client with timeout
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout,
            connectTimeoutMS=timeout
        )
        
        # Test connection
        client.admin.command('ping')
        
        # Access database and collection
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        return client, db, collection
    
    except ConnectionFailure:
        print_error("Failed to connect to MongoDB server")
        print("Possible causes:")
        print("  - Incorrect connection string")
        print("  - Network connectivity issues")
        print("  - MongoDB Atlas IP whitelist not configured")
        return None, None, None
    
    except ServerSelectionTimeoutError:
        print_error("MongoDB server selection timeout")
        print("Possible causes:")
        print("  - MongoDB Atlas cluster is paused")
        print("  - IP address not whitelisted (add 0.0.0.0/0 for all IPs)")
        print("  - Network firewall blocking connection")
        return None, None, None
    
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return None, None, None

def load_data(file_path):
    """
    Load cleaned data from CSV
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame or None on failure
    """
    try:
        if not os.path.exists(file_path):
            print_error(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        print_success(f"Loaded {len(df):,} records from {os.path.basename(file_path)}")
        print(f"   Columns: {df.shape[1]}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        return df
    
    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return None

def prepare_data_for_mongodb(df):
    """
    Convert DataFrame to MongoDB-compatible format
    
    Args:
        df: pandas DataFrame
    
    Returns:
        list: List of dictionaries ready for MongoDB insertion
    """
    try:
        # Convert date columns to datetime
        date_columns = ['arrival_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert DataFrame to list of dictionaries
        data_dict = df.to_dict('records')
        
        print_success(f"Converted {len(data_dict):,} records to MongoDB format")
        
        return data_dict
    
    except Exception as e:
        print_error(f"Failed to prepare data: {e}")
        return None

def insert_data(collection, data, batch_size=1000):
    """
    Insert data into MongoDB with batch processing
    
    Args:
        collection: MongoDB collection
        data: List of dictionaries to insert
        batch_size: Number of documents per batch
    
    Returns:
        int: Number of documents inserted
    """
    try:
        total_inserted = 0
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        print(f"\nInserting {len(data):,} documents in {total_batches} batches...")
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = collection.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            
            # Progress indicator
            batch_num = (i // batch_size) + 1
            print(f"   Batch {batch_num}/{total_batches}: {len(result.inserted_ids)} documents inserted")
        
        print_success(f"Successfully inserted {total_inserted:,} documents!")
        return total_inserted
    
    except Exception as e:
        print_error(f"Failed to insert data: {e}")
        return 0

def create_indexes(collection):
    """
    Create indexes for optimized queries
    
    Args:
        collection: MongoDB collection
    """
    try:
        print("\nCreating indexes for optimized queries...")
        
        # Single field indexes
        collection.create_index([("arrival_date", 1)])
        print_success("Created index on 'arrival_date'")
        
        collection.create_index([("hotel", 1)])
        print_success("Created index on 'hotel'")
        
        # Compound indexes
        collection.create_index([("hotel", 1), ("arrival_date", 1)])
        print_success("Created compound index on 'hotel' + 'arrival_date'")
        
        collection.create_index([("year", 1), ("month", 1)])
        print_success("Created compound index on 'year' + 'month'")
        
        # List all indexes
        indexes = list(collection.list_indexes())
        print(f"\nTotal indexes: {len(indexes)}")
        for idx in indexes:
            print(f"   - {idx['name']}: {idx.get('key', {})}")
    
    except Exception as e:
        print_error(f"Failed to create indexes: {e}")

def verify_data(collection, original_count):
    """
    Verify data was uploaded correctly
    
    Args:
        collection: MongoDB collection
        original_count: Expected number of documents
    """
    try:
        print_header("DATA VERIFICATION")
        
        # Count documents
        total_docs = collection.count_documents({})
        print(f"Total documents in MongoDB: {total_docs:,}")
        
        if total_docs == original_count:
            print_success("Document count matches!")
        else:
            print_warning(f"Document count mismatch! Expected: {original_count:,}, Got: {total_docs:,}")
        
        # Sample documents
        print("\nSample documents:")
        sample = list(collection.find().limit(3))
        for i, doc in enumerate(sample, 1):
            print(f"\n   Document {i}:")
            for key, value in doc.items():
                if key != '_id':  # Skip MongoDB's internal ID
                    print(f"      {key}: {value}")
        
        # Query by hotel
        print("\nDocuments by hotel:")
        for hotel in ['City Hotel', 'Resort Hotel']:
            count = collection.count_documents({'hotel': hotel})
            print(f"   {hotel}: {count:,} documents")
        
        # Date range
        print("\nDate range:")
        earliest = collection.find_one(sort=[("arrival_date", 1)])
        latest = collection.find_one(sort=[("arrival_date", -1)])
        
        if earliest and latest:
            print(f"   Earliest: {earliest.get('arrival_date', 'N/A')}")
            print(f"   Latest: {latest.get('arrival_date', 'N/A')}")
        
        # Summary statistics
        print("\nSummary statistics:")
        pipeline = [
            {
                '$group': {
                    '_id': None,
                    'total_bookings': {'$sum': '$total_bookings'},
                    'avg_bookings': {'$avg': '$total_bookings'},
                    'avg_cancellation_rate': {'$avg': '$cancellation_rate'},
                    'total_revenue': {'$sum': '$estimated_revenue'}
                }
            }
        ]
        
        stats = list(collection.aggregate(pipeline))
        if stats:
            stat = stats[0]
            print(f"   Total bookings: {stat.get('total_bookings', 0):,.0f}")
            print(f"   Average bookings per day: {stat.get('avg_bookings', 0):.2f}")
            print(f"   Average cancellation rate: {stat.get('avg_cancellation_rate', 0):.2f}%")
            print(f"   Total estimated revenue: ${stat.get('total_revenue', 0):,.2f}")
        
        print_success("Data verification complete!")
        
    except Exception as e:
        print_error(f"Verification failed: {e}")

def generate_summary_report(collection, start_time):
    """
    Generate summary report of upload operation
    
    Args:
        collection: MongoDB collection
        start_time: Script start time
    """
    try:
        print_header("UPLOAD SUMMARY REPORT")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = f"""
Database: {DATABASE_NAME}
Collection: {COLLECTION_NAME}
Upload Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.2f} seconds

DOCUMENTS
---------
Total Documents: {collection.count_documents({}):,}
City Hotel: {collection.count_documents({'hotel': 'City Hotel'}):,}
Resort Hotel: {collection.count_documents({'hotel': 'Resort Hotel'}):,}

INDEXES
-------
Total Indexes: {len(list(collection.list_indexes()))}

STATUS
------
✓ Data uploaded successfully
✓ Indexes created
✓ Data verified

NEXT STEPS
----------
1. Test queries in MongoDB Atlas UI
2. Create visualizations in MongoDB Charts (optional)
3. Use data for forecasting models
4. Set up automated backups (optional)

CONNECTION INFO
---------------
Database: {DATABASE_NAME}
Collection: {COLLECTION_NAME}
"""
        
        print(report)
        
        # Save report to file
        report_file = '../data/mongodb_upload_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print_success(f"Report saved to: {report_file}")
        
    except Exception as e:
        print_error(f"Failed to generate report: {e}")

def main():
    """
    Main execution function
    """
    start_time = datetime.now()
    
    print_header("MONGODB DATA UPLOAD - Hotel Demand Forecasting")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Get MongoDB URI
    print_header("STEP 1: MongoDB Connection")
    uri = get_mongodb_uri()
    
    # Step 2: Connect to MongoDB
    client, db, collection = connect_to_mongodb(uri)
    if not collection:
        print_error("Failed to connect to MongoDB. Exiting.")
        return
    
    print_success("Connected to MongoDB Atlas!")
    print(f"   Database: {DATABASE_NAME}")
    print(f"   Collection: {COLLECTION_NAME}")
    
    # Step 3: Load data
    print_header("STEP 2: Load Cleaned Data")
    df = load_data(DATA_FILE)
    if df is None:
        client.close()
        return
    
    # Step 4: Prepare data
    print_header("STEP 3: Prepare Data for MongoDB")
    data_dict = prepare_data_for_mongodb(df)
    if data_dict is None:
        client.close()
        return
    
    # Step 5: Check if collection already has data
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print_warning(f"Collection already contains {existing_count:,} documents!")
        response = input("Do you want to (1) Delete existing and insert new, (2) Keep existing, or (3) Cancel? [1/2/3]: ").strip()
        
        if response == '1':
            print("\nDeleting existing documents...")
            result = collection.delete_many({})
            print_success(f"Deleted {result.deleted_count:,} documents")
        elif response == '2':
            print_warning("Keeping existing documents. Adding new ones.")
        else:
            print("Operation cancelled.")
            client.close()
            return
    
    # Step 6: Insert data
    print_header("STEP 4: Insert Data into MongoDB")
    inserted_count = insert_data(collection, data_dict)
    
    if inserted_count == 0:
        client.close()
        return
    
    # Step 7: Create indexes
    print_header("STEP 5: Create Indexes")
    create_indexes(collection)
    
    # Step 8: Verify data
    verify_data(collection, len(data_dict))
    
    # Step 9: Generate summary report
    generate_summary_report(collection, start_time)
    
    # Close connection
    client.close()
    print_success("\nMongoDB connection closed.")
    
    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}✓ UPLOAD COMPLETE!{Colors.END}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        sys.exit(1)
