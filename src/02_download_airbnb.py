"""
InsideAirbnb European Data Downloader
======================================
Downloads Airbnb listings, calendar, and reviews data for major European cities
for hostel demand forecasting analysis.

Dataset: InsideAirbnb (http://insideairbnb.com/get-the-data/)
Cities: Amsterdam, Barcelona, Berlin, Paris, Rome, Prague

Author: Data Analysis Team
Date: October 24, 2025
"""

import requests
import pandas as pd
import os
import time
from pathlib import Path
import gzip
import shutil

# Configuration
BASE_URL = "http://data.insideairbnb.com"
# Try multiple data dates (InsideAirbnb updates quarterly)
DATA_DATES = ["2024-09-20", "2024-06-20", "2024-03-20", "2023-12-20", "2023-09-20"]
OUTPUT_DIR = "data/raw/airbnb"

# European cities for hostel demand analysis
CITIES = {
    'amsterdam': {
        'path': 'netherlands/north-holland/amsterdam',
        'country': 'Netherlands'
    },
    'barcelona': {
        'path': 'spain/catalonia/barcelona',
        'country': 'Spain'
    },
    'berlin': {
        'path': 'germany/berlin/berlin',
        'country': 'Germany'
    },
    'paris': {
        'path': 'france/ile-de-france/paris',
        'country': 'France'
    },
    'rome': {
        'path': 'italy/lazio/rome',
        'country': 'Italy'
    },
    'prague': {
        'path': 'czech-republic/prague/prague',
        'country': 'Czech Republic'
    }
}

# Dataset types to download
DATASETS = {
    'listings': 'listings.csv.gz',
    'calendar': 'calendar.csv.gz',
    'reviews': 'reviews.csv.gz'
}

def create_directory_structure():
    """Create directory structure for downloaded data"""
    print("\n" + "="*80)
    print("📁 CREATING DIRECTORY STRUCTURE")
    print("="*80)
    
    # Create main airbnb directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {OUTPUT_DIR}/")
    
    # Create subdirectories for each city
    for city in CITIES.keys():
        city_dir = os.path.join(OUTPUT_DIR, city)
        Path(city_dir).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {city_dir}/")
    
    print("\n✨ Directory structure ready!")

def download_file(url, output_path, file_type="file", show_url=True):
    """
    Download a file from URL and save to output path
    
    Args:
        url (str): URL to download from
        output_path (str): Local path to save file
        file_type (str): Type of file for display purposes
        show_url (bool): Whether to print the URL
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if show_url:
            print(f"   📥 Downloading {file_type}...")
            print(f"      URL: {url}")
        
        response = requests.get(url, stream=True, timeout=60)
        
        if response.status_code == 200:
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"   ✅ Downloaded: {os.path.basename(output_path)} ({file_size:.2f} MB)")
            
            return True
        else:
            print(f"   ❌ Failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def decompress_gz_file(gz_path):
    """
    Decompress a .gz file to .csv
    
    Args:
        gz_path (str): Path to .gz file
    
    Returns:
        str: Path to decompressed CSV file
    """
    csv_path = gz_path.replace('.gz', '')
    
    try:
        print(f"   📦 Decompressing {os.path.basename(gz_path)}...")
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove .gz file after decompression
        os.remove(gz_path)
        
        # Get CSV file size
        csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        print(f"   ✅ Decompressed: {os.path.basename(csv_path)} ({csv_size:.2f} MB)")
        
        return csv_path
        
    except Exception as e:
        print(f"   ❌ Decompression error: {e}")
        return None

def download_city_data(city_name, city_info):
    """
    Download all datasets for a specific city (tries multiple dates)
    
    Args:
        city_name (str): Name of the city
        city_info (dict): Dictionary with city path and country
    
    Returns:
        dict: Summary of downloaded files
    """
    print("\n" + "="*80)
    print(f"🏙️  DOWNLOADING DATA FOR: {city_name.upper()} ({city_info['country']})")
    print("="*80)
    
    city_path = city_info['path']
    city_dir = os.path.join(OUTPUT_DIR, city_name)
    
    summary = {
        'city': city_name,
        'country': city_info['country'],
        'downloaded': [],
        'failed': [],
        'data_date': None
    }
    
    # Try each dataset type
    for dataset_name, dataset_file in DATASETS.items():
        print(f"\n📊 {dataset_name.upper()} Data:")
        
        downloaded = False
        
        # Try multiple data dates
        for data_date in DATA_DATES:
            # Construct URL
            url = f"{BASE_URL}/{city_path}/{data_date}/data/{dataset_file}"
            
            # Output path
            output_path = os.path.join(city_dir, dataset_file)
            
            # Try to download (suppress URL printing after first attempt)
            show_url = (data_date == DATA_DATES[0])
            if not show_url:
                print(f"   ⏳ Trying alternate date: {data_date}...")
            
            success = download_file(url, output_path, dataset_name, show_url=show_url)
            
            if success:
                # Decompress .gz file
                csv_path = decompress_gz_file(output_path)
                if csv_path:
                    summary['downloaded'].append(dataset_name)
                    if summary['data_date'] is None:
                        summary['data_date'] = data_date
                    downloaded = True
                    break
        
        if not downloaded:
            print(f"   ❌ Failed: Could not find data for any date")
            summary['failed'].append(dataset_name)
        
        # Be nice to the server
        time.sleep(1)
    
    return summary

def verify_downloaded_data(city_name):
    """
    Verify and display basic info about downloaded data
    
    Args:
        city_name (str): Name of the city
    """
    print(f"\n   🔍 Verifying {city_name} data...")
    
    city_dir = os.path.join(OUTPUT_DIR, city_name)
    
    for dataset_name in DATASETS.keys():
        csv_file = os.path.join(city_dir, f"{dataset_name}.csv")
        
        if os.path.exists(csv_file):
            try:
                # Read first few rows to get basic info
                df = pd.read_csv(csv_file, nrows=5)
                total_rows = len(pd.read_csv(csv_file))
                
                print(f"      ✅ {dataset_name}.csv: {total_rows:,} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"      ⚠️  {dataset_name}.csv: Error reading file - {e}")
        else:
            print(f"      ❌ {dataset_name}.csv: File not found")

def download_all_cities():
    """Download data for all configured cities"""
    print("\n" + "="*80)
    print("🌍 INSIDEAIRBNB EUROPEAN DATA DOWNLOADER")
    print("="*80)
    print(f"\nTarget Cities: {', '.join([c.title() for c in CITIES.keys()])}")
    print(f"Data Dates (trying): {', '.join(DATA_DATES)}")
    print(f"Datasets: listings, calendar, reviews")
    
    # Create directories
    create_directory_structure()
    
    # Track overall progress
    all_summaries = []
    start_time = time.time()
    
    # Download data for each city
    for city_name, city_info in CITIES.items():
        summary = download_city_data(city_name, city_info)
        all_summaries.append(summary)
        
        # Verify downloaded data
        verify_downloaded_data(city_name)
        
        print(f"\n✨ {city_name.title()} download complete!")
        if summary['data_date']:
            print(f"   📅 Data date: {summary['data_date']}")
        print(f"   ✅ Successfully downloaded: {len(summary['downloaded'])} datasets")
        if summary['failed']:
            print(f"   ❌ Failed to download: {', '.join(summary['failed'])}")
    
    # Calculate total time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    # Print final summary
    print("\n" + "="*80)
    print("✨ DOWNLOAD COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Overall Summary:")
    print(f"   Total Cities: {len(CITIES)}")
    print(f"   Time Elapsed: {minutes}m {seconds}s")
    
    total_success = sum(len(s['downloaded']) for s in all_summaries)
    total_failed = sum(len(s['failed']) for s in all_summaries)
    
    print(f"\n   ✅ Successfully downloaded: {total_success} datasets")
    if total_failed > 0:
        print(f"   ❌ Failed downloads: {total_failed} datasets")
    
    print(f"\n📂 Data saved to: {OUTPUT_DIR}/")
    
    # Print city-by-city summary
    print(f"\n🏙️  City Summary:")
    for summary in all_summaries:
        status = "✅" if len(summary['failed']) == 0 else "⚠️"
        date_info = f" (data: {summary['data_date']})" if summary['data_date'] else ""
        print(f"   {status} {summary['city'].title()} ({summary['country']}): {len(summary['downloaded'])}/{len(DATASETS)} datasets{date_info}")
    
    print("\n" + "="*80)
    print("🎯 Next Steps:")
    print("   1. Review downloaded data in data/airbnb/ folder")
    print("   2. Run: python 03_explore_airbnb_data.py")
    print("   3. Combine hotel + Airbnb data for comprehensive analysis")
    print("="*80)

def main():
    """Main execution function"""
    try:
        download_all_cities()
        
        print("\n✅ Script completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
