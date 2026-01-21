#!/usr/bin/env python3
"""
Download Form 4 filings from SEC API and upload to Google Cloud Storage.
Organizes files by symbol and year: SYMBOL/YEAR/SYMBOL_YYYYMMDD_HASH.json
"""

import os
import sys
import json
import hashlib
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from google.cloud import storage
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables
load_dotenv('env')

# Configuration
SEC_API_KEY = os.getenv("SEC_API_KEY")
SEC_API_ENDPOINT = os.getenv("SEC_API_ENDPOINT", "https://api.sec-api.io/insider-trading")
TRACKED_SYMBOLS_STR = os.getenv("TRACKED_SYMBOLS", "TSLA,META,MSFT,AMZN,NFLX,AAPL,NVDA,GOOGL")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "form-4-filings")

# Date range
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# API pagination
MAX_RESULTS_PER_QUERY = 50  # SEC API limit
MAX_RESULTS_TOTAL = 10000   # SEC API max

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Parallel upload configuration
MAX_CONCURRENT_UPLOADS = int(os.getenv("MAX_CONCURRENT_UPLOADS", "20"))  # Parallel uploads
MAX_CONCURRENT_CHECKS = int(os.getenv("MAX_CONCURRENT_CHECKS", "50"))  # Parallel existence checks

if not SEC_API_KEY:
    print("❌ Error: SEC_API_KEY not found in environment variables")
    sys.exit(1)

# Parse tracked symbols
TRACKED_SYMBOLS = [s.strip() for s in TRACKED_SYMBOLS_STR.split(",") if s.strip()]

# Initialize GCS client
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print(f"✅ GCS client initialized")
    print(f"   Bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    print(f"❌ Failed to initialize GCS client: {e}")
    sys.exit(1)


def generate_hash(accession_no: str) -> str:
    """Generate 8-character hash from accession number."""
    # Use first 8 characters of MD5 hash
    hash_obj = hashlib.md5(accession_no.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def query_sec_api(query_params: Dict, retry_count: int = 0) -> Optional[Dict]:
    """Make a POST request to the SEC API with retry logic."""
    headers = {
        "Authorization": SEC_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            SEC_API_ENDPOINT,
            headers=headers,
            json=query_params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if retry_count < MAX_RETRIES:
            print(f"   ⚠️  API request failed (attempt {retry_count + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY * (retry_count + 1))  # Exponential backoff
            return query_sec_api(query_params, retry_count + 1)
        else:
            print(f"   ❌ API request failed after {MAX_RETRIES} retries: {e}")
            return None


def get_all_filings_for_symbol(symbol: str, start_date: str, end_date: str) -> List[Dict]:
    """Get all filings for a symbol with pagination."""
    all_filings = []
    from_offset = 0
    
    print(f"   📥 Fetching filings for {symbol}...")
    
    while True:
        query = f"issuer.tradingSymbol:{symbol} AND periodOfReport:[{start_date} TO {end_date}]"
        query_params = {
            "query": query,
            "from": from_offset,
            "size": MAX_RESULTS_PER_QUERY,
            "sort": [{"periodOfReport": {"order": "asc"}}]
        }
        
        result = query_sec_api(query_params)
        
        if not result:
            print(f"   ⚠️  Failed to fetch batch starting at offset {from_offset}")
            break
        
        transactions = result.get("transactions", [])
        if not transactions:
            break
        
        all_filings.extend(transactions)
        
        total = result.get("total", {})
        total_value = total.get("value", 0)
        total_relation = total.get("relation", "eq")
        
        print(f"      Fetched {len(transactions)} filings (total so far: {len(all_filings)})")
        
        # Check if we've got all results
        if total_relation == "eq" and len(all_filings) >= total_value:
            print(f"      ✅ Retrieved all {total_value} filings")
            break
        
        # Check if we've hit the max
        if len(all_filings) >= MAX_RESULTS_TOTAL:
            print(f"      ⚠️  Hit max results limit ({MAX_RESULTS_TOTAL}), stopping")
            break
        
        # Check if this was the last page
        if len(transactions) < MAX_RESULTS_PER_QUERY:
            break
        
        from_offset += MAX_RESULTS_PER_QUERY
    
    return all_filings


def get_gcs_path(symbol: str, period_of_report: str, accession_no: str) -> str:
    """Generate GCS path for a filing."""
    # Extract year from period_of_report (YYYY-MM-DD format)
    year = period_of_report[:4]
    
    # Generate hash from accession number
    hash_str = generate_hash(accession_no)
    
    # Format date as YYYYMMDD
    date_str = period_of_report.replace("-", "")
    
    # Filename: SYMBOL_YYYYMMDD_HASH.json
    filename = f"{symbol}_{date_str}_{hash_str}.json"
    
    # Path: form-4-filings/SYMBOL/YEAR/SYMBOL_YYYYMMDD_HASH.json
    gcs_path = f"{GCS_BASE_PATH}/{symbol}/{year}/{filename}"
    
    return gcs_path


def file_exists_in_gcs(gcs_path: str) -> bool:
    """Check if a file already exists in GCS."""
    try:
        blob = bucket.blob(gcs_path)
        return blob.exists()
    except Exception as e:
        print(f"      ⚠️  Error checking existence for {gcs_path}: {e}")
        return False  # Assume doesn't exist, will try to upload


def upload_filing_to_gcs(filing: Dict, gcs_path: str, retry_count: int = 0) -> bool:
    """Upload a filing to GCS with retry logic."""
    try:
        # Convert to JSON string
        json_str = json.dumps(filing, ensure_ascii=False, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        # Upload to GCS
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(json_bytes, content_type='application/json')
        
        return True
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"      ⚠️  Upload failed (attempt {retry_count + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY * (retry_count + 1))
            return upload_filing_to_gcs(filing, gcs_path, retry_count + 1)
        else:
            print(f"      ❌ Upload failed after {MAX_RETRIES} retries: {e}")
            return False


def check_file_exists(gcs_path: str) -> bool:
    """Check if a file exists in GCS (for parallel batch checking)."""
    return file_exists_in_gcs(gcs_path)


def upload_filing_only(filing: Dict, gcs_path: str) -> bool:
    """Upload a filing to GCS (assumes existence already checked)."""
    return upload_filing_to_gcs(filing, gcs_path)


def process_symbol(symbol: str) -> Dict:
    """Process all filings for a single symbol with parallel uploads."""
    print(f"\n{'='*80}")
    print(f"Processing {symbol}")
    print(f"{'='*80}")
    
    # Get all filings
    filings = get_all_filings_for_symbol(symbol, START_DATE, END_DATE)
    
    if not filings:
        print(f"   ⚠️  No filings found for {symbol}")
        return {
            "symbol": symbol,
            "total_found": 0,
            "uploaded": 0,
            "skipped": 0,
            "failed": 0
        }
    
    print(f"   📊 Found {len(filings)} total filings")
    
    # Prepare filing data with GCS paths
    filing_data = []
    for i, filing in enumerate(filings):
        accession_no = filing.get("accessionNo", "")
        period_of_report = filing.get("periodOfReport", "")
        
        if not accession_no or not period_of_report:
            continue
        
        gcs_path = get_gcs_path(symbol, period_of_report, accession_no)
        filing_data.append({
            "filing": filing,
            "gcs_path": gcs_path,
            "filename": gcs_path.split('/')[-1],
            "index": i + 1
        })
    
    if not filing_data:
        print(f"   ⚠️  No valid filings to process")
        return {
            "symbol": symbol,
            "total_found": len(filings),
            "uploaded": 0,
            "skipped": 0,
            "failed": len(filings)
        }
    
    # Step 1: Batch check file existence in parallel
    print(f"   🔍 Checking existing files (max {MAX_CONCURRENT_CHECKS} concurrent)...")
    start_time = time.time()
    existing_files = set()
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHECKS) as executor:
        future_to_path = {
            executor.submit(check_file_exists, data["gcs_path"]): data
            for data in filing_data
        }
        
        checked = 0
        for future in as_completed(future_to_path):
            data = future_to_path[future]
            try:
                exists = future.result()
                if exists:
                    existing_files.add(data["gcs_path"])
                checked += 1
                if checked % 100 == 0 or checked == len(filing_data):
                    print(f"      Checked {checked}/{len(filing_data)} files...")
            except Exception as e:
                print(f"      ⚠️  Error checking {data['filename']}: {e}")
    
    check_elapsed = time.time() - start_time
    to_upload = [data for data in filing_data if data["gcs_path"] not in existing_files]
    
    print(f"   ✅ Existence check complete in {check_elapsed:.1f}s")
    print(f"      Skipped (exists): {len(existing_files)}")
    print(f"      To upload: {len(to_upload)}")
    
    # Step 2: Upload missing files in parallel
    if not to_upload:
        print(f"   ℹ️  All files already exist, nothing to upload")
        return {
            "symbol": symbol,
            "total_found": len(filings),
            "uploaded": 0,
            "skipped": len(existing_files),
            "failed": 0
        }
    
    print(f"   🚀 Starting parallel uploads (max {MAX_CONCURRENT_UPLOADS} concurrent)...")
    stats = {
        "symbol": symbol,
        "total_found": len(filings),
        "uploaded": 0,
        "skipped": len(existing_files),
        "failed": 0
    }
    stats_lock = Lock()
    
    upload_start = time.time()
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_UPLOADS) as executor:
        future_to_data = {
            executor.submit(upload_filing_only, data["filing"], data["gcs_path"]): data
            for data in to_upload
        }
        
        for future in as_completed(future_to_data):
            data = future_to_data[future]
            try:
                success = future.result()
                with stats_lock:
                    if success:
                        stats["uploaded"] += 1
                    else:
                        stats["failed"] += 1
                    completed += 1
                    
                    # Print progress every 50 files or at completion
                    if completed % 50 == 0 or completed == len(to_upload):
                        elapsed = time.time() - upload_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = len(to_upload) - completed
                        eta = remaining / rate if rate > 0 else 0
                        
                        print(f"   [{completed}/{len(to_upload)}] "
                              f"Uploaded: {stats['uploaded']}, "
                              f"Failed: {stats['failed']} | "
                              f"Rate: {rate:.1f}/s, ETA: {eta:.0f}s")
            except Exception as e:
                with stats_lock:
                    stats["failed"] += 1
                    completed += 1
                    print(f"   ⚠️  Exception uploading {data['filename']}: {e}")
    
    upload_elapsed = time.time() - upload_start
    total_elapsed = time.time() - start_time
    
    print(f"   ✅ Upload complete in {upload_elapsed:.1f}s ({len(to_upload)/upload_elapsed:.1f} files/sec)")
    print(f"   ⏱️  Total time: {total_elapsed:.1f}s")
    
    return stats


def main():
    """Main function."""
    print("="*80)
    print("Form 4 Filings Download to GCS")
    print("="*80)
    print(f"Bucket: gs://{GCS_BUCKET_NAME}")
    print(f"Base Path: {GCS_BASE_PATH}")
    print(f"Symbols: {', '.join(TRACKED_SYMBOLS)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Max Concurrent Uploads: {MAX_CONCURRENT_UPLOADS}")
    print()
    
    all_stats = []
    
    # Process each symbol
    for symbol in TRACKED_SYMBOLS:
        stats = process_symbol(symbol)
        all_stats.append(stats)
        
        # Print summary for this symbol
        print(f"\n   📊 {symbol} Summary:")
        print(f"      Total found: {stats['total_found']}")
        print(f"      Uploaded: {stats['uploaded']}")
        print(f"      Skipped (already exists): {stats['skipped']}")
        print(f"      Failed: {stats['failed']}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_found = sum(s["total_found"] for s in all_stats)
    total_uploaded = sum(s["uploaded"] for s in all_stats)
    total_skipped = sum(s["skipped"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    
    print(f"\n{'Symbol':<8} {'Found':<8} {'Uploaded':<10} {'Skipped':<10} {'Failed':<8}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['symbol']:<8} {stats['total_found']:<8} {stats['uploaded']:<10} "
              f"{stats['skipped']:<10} {stats['failed']:<8}")
    
    print("-" * 80)
    print(f"{'TOTAL':<8} {total_found:<8} {total_uploaded:<10} {total_skipped:<10} {total_failed:<8}")
    
    print(f"\n✅ Download complete!")
    print(f"   Files uploaded to: gs://{GCS_BUCKET_NAME}/{GCS_BASE_PATH}/")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
