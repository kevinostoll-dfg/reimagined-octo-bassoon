#!/usr/bin/env python3
"""
FMP Earnings Transcript Downloader

This script downloads earnings call transcripts from Financial Modeling Prep API
for specified stock symbols over the past 10 years and uploads them to GCS.

GCS path: earnings-announcement-transcripts/<symbol>/<year>.<quarter>.json
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock, Semaphore
from typing import Tuple

import requests
from dotenv import load_dotenv
from google.cloud import storage


# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    raise ValueError("FMP_API_KEY environment variable not set. Please set it in .env file or export it.")

BASE_URL = "https://financialmodelingprep.com/api/v3/earning_call_transcript"

# GCS Configuration
GCS_BUCKET_NAME = "blacksmith-sec-filings"
GCS_BASE_PATH = "earnings-announcement-transcripts"

# Stock symbols
SYMBOLS = ["AAPL", "META", "MSFT", "TSLA", "GOOGL", "AVGO", "NVDA", "AMZN", "NFLX"]

# Calculate year range (past 10 years from current date)
current_year = datetime.now().year
START_YEAR = current_year - 10
END_YEAR = current_year

# Quarters
QUARTERS = [1, 2, 3, 4]

# Initialize GCS client (uses Application Default Credentials)
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Threading configuration
MAX_WORKERS = 50  # High concurrency for speed
API_SEMAPHORE = Semaphore(15)  # Limit concurrent API calls to avoid rate limits
stats_lock = Lock()  # Thread-safe counter

# Retry configuration
MAX_RETRIES = 5  # Maximum number of retries for 429 errors
INITIAL_RETRY_DELAY = 2  # Initial delay in seconds (exponential backoff)

# Statistics
stats = {
    "total": 0,
    "successful": 0,
    "skipped": 0,
    "failed": 0,
    "retries": 0
}

def get_gcs_path(symbol: str, year: int, quarter: int) -> str:
    """Generate GCS path for a transcript."""
    return f"{GCS_BASE_PATH}/{symbol}/{year}.{quarter}.json"

def file_exists_in_gcs(gcs_path: str) -> bool:
    """Check if file already exists in GCS."""
    blob = bucket.blob(gcs_path)
    return blob.exists()

def upload_to_gcs(gcs_path: str, data: dict) -> bool:
    """Upload JSON data to GCS."""
    try:
        blob = bucket.blob(gcs_path)
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        blob.upload_from_string(json_str, content_type="application/json")
        return True
    except Exception as e:
        print(f"✗ GCS upload error for {gcs_path}: {str(e)}")
        return False

def download_and_upload_transcript(symbol: str, year: int, quarter: int) -> Tuple[bool, str]:
    """
    Download earnings transcript and upload to GCS with retry logic for rate limits.
    
    Args:
        symbol: Stock ticker symbol
        year: Year of the earnings call
        quarter: Quarter number (1-4)
        
    Returns:
        Tuple of (success: bool, status_message: str)
    """
    gcs_path = get_gcs_path(symbol, year, quarter)
    
    # Check if file already exists in GCS
    if file_exists_in_gcs(gcs_path):
        with stats_lock:
            stats["total"] += 1
            stats["skipped"] += 1
        return True, "skipped"
    
    url = f"{BASE_URL}/{symbol}"
    params = {
        "year": year,
        "quarter": quarter,
        "apikey": API_KEY
    }
    
    # Retry logic with exponential backoff for 429 errors
    for attempt in range(MAX_RETRIES + 1):
        # Acquire semaphore for API call
        with API_SEMAPHORE:
            try:
                response = requests.get(url, params=params, timeout=30)
                
                # Handle 429 rate limit errors with retry
                if response.status_code == 429:
                    if attempt < MAX_RETRIES:
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                        with stats_lock:
                            stats["retries"] += 1
                        time.sleep(delay)
                        continue  # Retry
                    else:
                        with stats_lock:
                            stats["total"] += 1
                            stats["failed"] += 1
                        return False, "rate_limit_exceeded"
                
                response.raise_for_status()
                data = response.json()
                
                if not data or len(data) == 0:
                    with stats_lock:
                        stats["total"] += 1
                        stats["failed"] += 1
                    return False, "no_data"
                
                # Upload to GCS (must succeed)
                if upload_to_gcs(gcs_path, data):
                    with stats_lock:
                        stats["total"] += 1
                        stats["successful"] += 1
                    return True, "uploaded"
                else:
                    with stats_lock:
                        stats["total"] += 1
                        stats["failed"] += 1
                    return False, "upload_failed"
                    
            except requests.exceptions.HTTPError as e:
                # Non-429 HTTP errors - don't retry
                if e.response.status_code != 429:
                    with stats_lock:
                        stats["total"] += 1
                        stats["failed"] += 1
                    return False, f"http_error_{e.response.status_code}"
                # 429 handled above, but just in case
                if attempt < MAX_RETRIES:
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    with stats_lock:
                        stats["retries"] += 1
                    time.sleep(delay)
                    continue
                else:
                    with stats_lock:
                        stats["total"] += 1
                        stats["failed"] += 1
                    return False, "rate_limit_exceeded"
            except requests.exceptions.RequestException as e:
                # Network errors - retry
                if attempt < MAX_RETRIES:
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    with stats_lock:
                        stats["retries"] += 1
                    time.sleep(delay)
                    continue
                else:
                    with stats_lock:
                        stats["total"] += 1
                        stats["failed"] += 1
                    return False, f"network_error: {str(e)}"
            except Exception as e:
                with stats_lock:
                    stats["total"] += 1
                    stats["failed"] += 1
                return False, f"error: {str(e)}"
    
    # Should never reach here, but just in case
    with stats_lock:
        stats["total"] += 1
        stats["failed"] += 1
    return False, "max_retries_exceeded"

def process_transcript(args: Tuple[str, int, int]) -> Tuple[str, int, int, bool, str]:
    """Wrapper function for parallel processing."""
    symbol, year, quarter = args
    success, status = download_and_upload_transcript(symbol, year, quarter)
    return (symbol, year, quarter, success, status)

def main():
    """Main execution function."""
    print("=" * 80)
    print("FMP Earnings Transcript Downloader & GCS Uploader")
    print("=" * 80)
    print(f"\n🎯 Symbols: {', '.join(SYMBOLS)}")
    print(f"📅 Year Range: {START_YEAR} - {END_YEAR}")
    print(f"📊 Quarters: Q1, Q2, Q3, Q4")
    print(f"☁️  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 GCS Path: {GCS_BASE_PATH}/<symbol>/<year>.<quarter>.json")
    print(f"⚡ Max Workers: {MAX_WORKERS}")
    print(f"🔒 API Concurrency Limit: {API_SEMAPHORE._value}")
    print(f"🔄 Max Retries: {MAX_RETRIES} (with exponential backoff)")
    print(f"{'=' * 80}\n")
    
    # Generate all tasks
    tasks = []
    for symbol in SYMBOLS:
        for year in range(START_YEAR, END_YEAR + 1):
            for quarter in QUARTERS:
                tasks.append((symbol, year, quarter))
    
    print(f"📋 Total tasks: {len(tasks)}")
    print(f"🚀 Starting parallel processing...\n")
    
    start_time = time.time()
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_transcript, task): task 
            for task in tasks
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                symbol, year, quarter, success, status = future.result()
                if status == "skipped":
                    print(f"⏭️  [{completed}/{len(tasks)}] {symbol} {year}.{quarter} - Skipped (already exists)")
                elif success:
                    print(f"✅ [{completed}/{len(tasks)}] {symbol} {year}.{quarter} - Uploaded")
                else:
                    print(f"❌ [{completed}/{len(tasks)}] {symbol} {year}.{quarter} - Failed: {status}")
            except Exception as e:
                print(f"❌ [{completed}/{len(tasks)}] Exception: {str(e)}")
    
    # Summary
    elapsed_time = time.time() - start_time
    success_rate = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    print(f"\n{'=' * 80}")
    print("✅ PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"📊 Total Tasks: {stats['total']}")
    print(f"✅ Successful Uploads: {stats['successful']}")
    print(f"⏭️  Skipped (already exists): {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"🔄 Retries: {stats['retries']}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"⏱️  Time Elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"☁️  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 GCS Path: {GCS_BASE_PATH}/<symbol>/<year>.<quarter>.json")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()

