#!/usr/bin/env python3
"""
SEC 10-K Section Extractor & GCS Uploader

This script queries the SEC API to retrieve the last 10 10-K filings for specified tickers,
then extracts all sections from each filing and uploads them directly to GCS.

GCS path: sec-10k/<ticker>/<year>/<section>.json
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock, Semaphore
from typing import Dict, List, Optional, Tuple

import requests
from google.cloud import storage


# Configuration
API_KEY = "ef3ca09dd5d0b80bf2969fdd10e160c7b3e08c93af33fbc4110fa1910877a49c"
QUERY_API_URL = "https://api.sec-api.io"
EXTRACTOR_API_URL = "https://api.sec-api.io/extractor"

# Updated tickers
TICKERS = ["AAPL", "META", "MSFT", "AMZN", "NFLX", "GOOGL", "AVGO", "NVDA", "TSLA"]
NUM_FILINGS = 10

# GCS Configuration
GCS_BUCKET_NAME = "blacksmith-sec-filings"
GCS_BASE_PATH = "sec-10k"

# 10-K sections to extract
SECTIONS = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", 
    "7", "7A", "8", "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"
]

# Comprehensive section names for Form 10-K metadata (as of December 2025)
SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary"
}

# Initialize GCS client (uses Application Default Credentials)
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Threading configuration
MAX_WORKERS = 100  # High concurrency for parallel section processing
API_SEMAPHORE = Semaphore(20)  # Limit concurrent API calls to avoid rate limits
stats_lock = Lock()  # Thread-safe counter

# Retry configuration
MAX_RETRIES = 5  # Maximum number of retries for API and GCS operations
INITIAL_RETRY_DELAY = 1  # Initial delay in seconds (exponential backoff)
MAX_RETRY_DELAY = 60  # Maximum delay between retries

# Statistics
stats = {
    "total_sections": 0,
    "successful": 0,
    "skipped": 0,
    "failed": 0,
    "api_retries": 0,
    "gcs_retries": 0
}


def get_gcs_path(ticker: str, year: str, section: str) -> str:
    """Generate GCS path for a section."""
    return f"{GCS_BASE_PATH}/{ticker}/{year}/{section}.json"


def file_exists_in_gcs(gcs_path: str) -> bool:
    """Check if file already exists in GCS."""
    try:
        blob = bucket.blob(gcs_path)
        return blob.exists()
    except Exception as e:
        print(f"⚠️  Error checking GCS file existence for {gcs_path}: {e}")
        return False


def upload_to_gcs_with_retry(gcs_path: str, json_data: dict) -> bool:
    """
    Upload JSON data to GCS with retry logic. MUST succeed.
    
    Args:
        gcs_path: GCS path for the file
        json_data: JSON data to upload
        
    Returns:
        True if successful, raises exception if all retries fail
    """
    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            blob = bucket.blob(gcs_path)
            blob.upload_from_string(json_str, content_type="application/json")
            
            # Verify upload
            blob.reload()
            if not blob.exists():
                raise Exception(f"Upload verification failed for {gcs_path}")
            
            if attempt > 0:
                with stats_lock:
                    stats["gcs_retries"] += attempt
            return True
            
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                with stats_lock:
                    stats["gcs_retries"] += 1
                time.sleep(delay)
                continue
            else:
                # Final attempt failed - raise exception (GCS uploads MUST succeed)
                raise Exception(f"GCS upload failed after {MAX_RETRIES} retries: {str(e)}")


def query_10k_filings(ticker: str, limit: int = 10) -> List[Dict]:
    """
    Query the SEC API for the most recent 10-K filings for a given ticker.
    Excludes amendments (10-K/A) and NT (not timely) filings.
    
    Args:
        ticker: Stock ticker symbol
        limit: Number of filings to retrieve
        
    Returns:
        List of filing metadata dictionaries
    """
    # Exclude amendments and NT filings for more reliable results
    payload = {
        "query": f'ticker:{ticker} AND formType:"10-K" AND NOT formType:("10-K/A", "NT")',
        "from": "0",
        "size": str(limit),
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(QUERY_API_URL, json=payload, headers=headers, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {MAX_RETRIES} retries")
            
            response.raise_for_status()
            data = response.json()
            
            filings = data.get("filings", [])
            
            # Debug output for GOOGL to understand why only 1 filing
            if ticker == "GOOGL" and len(filings) < limit:
                total = data.get("total", {})
                print(f"    ⚠️  GOOGL query returned {len(filings)} filings (requested {limit})")
                print(f"    📊 API total count: {total}")
                if filings:
                    print(f"    📅 Filing years: {[f.get('periodOfReport', 'N/A')[:4] for f in filings]}")
                    print(f"    📄 Filing dates: {[f.get('filedAt', 'N/A')[:10] for f in filings]}")
            
            return filings
            
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                time.sleep(delay)
                continue
            else:
                print(f"❌ Error querying filings for {ticker} after {MAX_RETRIES} retries: {e}")
                return []
    
    return []


def extract_section(filing_url: str, section: str) -> Optional[str]:
    """
    Extract a specific section from a 10-K filing with retry logic and rate limiting.
    
    Args:
        filing_url: URL of the 10-K filing
        section: Section code (e.g., "1A", "7", etc.)
        
    Returns:
        Extracted section content as text, or None if extraction fails
    """
    params = {
        "url": filing_url,
        "item": section,
        "type": "text",
        "token": API_KEY
    }
    
    # Use semaphore to limit concurrent API calls
    with API_SEMAPHORE:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.get(EXTRACTOR_API_URL, params=params, timeout=60)
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < MAX_RETRIES:
                        delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                        with stats_lock:
                            stats["api_retries"] += 1
                        time.sleep(delay)
                        continue
                    else:
                        return None
                
                response.raise_for_status()
                
                # Check if response indicates processing
                content = response.text
                
                if content.lower().startswith("processing"):
                    if attempt < MAX_RETRIES:
                        delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                        time.sleep(delay)
                        continue
                    else:
                        return None
                
                return content
                
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    with stats_lock:
                        stats["api_retries"] += 1
                    time.sleep(delay)
                    continue
                else:
                    return None
    
    return None


def process_section(ticker: str, year: str, section: str, filing_url: str, 
                   filing_metadata: Dict) -> Tuple[bool, str]:
    """
    Extract a section and upload to GCS. This is the smallest granularity task.
    
    Args:
        ticker: Stock ticker symbol
        year: Year from periodOfReport
        section: Section code
        filing_url: URL of the 10-K filing
        filing_metadata: Metadata about the filing
        
    Returns:
        Tuple of (success: bool, status_message: str)
    """
    gcs_path = get_gcs_path(ticker, year, section)
    
    # Check if file already exists in GCS
    if file_exists_in_gcs(gcs_path):
        with stats_lock:
            stats["total_sections"] += 1
            stats["skipped"] += 1
        return True, "skipped"
    
    # Extract section content
    content = extract_section(filing_url, section)
    
    if not content:
        # Some sections may not exist in older filings (e.g., 1C, 9C, 16)
        # This is expected behavior - mark as skipped rather than failed
        with stats_lock:
            stats["total_sections"] += 1
            stats["skipped"] += 1
        return True, "section_not_found"
    
    # Prepare JSON data
    json_data = {
        "metadata": {
            "ticker": ticker,
            "company_name": filing_metadata.get("companyName", ""),
            "cik": filing_metadata.get("cik", ""),
            "accession_no": filing_metadata.get("accessionNo", ""),
            "form_type": filing_metadata.get("formType", ""),
            "filed_at": filing_metadata.get("filedAt", ""),
            "period_of_report": filing_metadata.get("periodOfReport", ""),
            "filing_url": filing_metadata.get("linkToFilingDetails", ""),
            "section": section,
            "section_name": SECTION_NAMES.get(section, ""),
            "extracted_at": datetime.now(timezone.utc).isoformat()
        },
        "content": content
    }
    
    # Upload to GCS (must succeed - will retry)
    try:
        upload_to_gcs_with_retry(gcs_path, json_data)
        with stats_lock:
            stats["total_sections"] += 1
            stats["successful"] += 1
        return True, "uploaded"
    except Exception as e:
        with stats_lock:
            stats["total_sections"] += 1
            stats["failed"] += 1
        return False, f"upload_failed: {str(e)}"


def process_section_wrapper(args: Tuple[str, str, str, str, Dict]) -> Tuple[str, str, str, bool, str]:
    """Wrapper function for parallel processing."""
    ticker, year, section, filing_url, filing_metadata = args
    success, status = process_section(ticker, year, section, filing_url, filing_metadata)
    return (ticker, year, section, success, status)


def main():
    """Main execution function."""
    print("=" * 80)
    print("SEC 10-K Section Extractor & GCS Uploader")
    print("=" * 80)
    print(f"\n🎯 Tickers: {', '.join(TICKERS)}")
    print(f"☁️  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 GCS Path: {GCS_BASE_PATH}/<ticker>/<year>/<section>.json")
    print(f"🔢 Filings per ticker: {NUM_FILINGS}")
    print(f"📋 Sections to extract: {len(SECTIONS)}")
    print(f"⚡ Max Workers: {MAX_WORKERS}")
    print(f"🔒 API Concurrency Limit: {API_SEMAPHORE._value}")
    print(f"🔄 Max Retries: {MAX_RETRIES} (with exponential backoff)")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    # Collect all tasks (ticker, year, section, filing_url, filing_metadata)
    all_tasks = []
    
    print("📊 Querying filings for all tickers...\n")
    for ticker in TICKERS:
        print(f"  Querying {ticker}...", end=" ")
        filings = query_10k_filings(ticker, NUM_FILINGS)
        print(f"Found {len(filings)} filings")
        
        if not filings:
            continue
        
        for filing in filings:
            period_of_report = filing.get("periodOfReport", "")
            filing_url = filing.get("linkToFilingDetails", "")
            
            if not period_of_report or not filing_url:
                continue
            
            # Extract year from periodOfReport (format: YYYY-MM-DD)
            year = period_of_report.split("-")[0]
            
            # Create a task for each section
            for section in SECTIONS:
                all_tasks.append((ticker, year, section, filing_url, filing))
    
    print(f"\n📋 Total tasks: {len(all_tasks)}")
    print(f"🚀 Starting parallel processing...\n")
    
    # Process all sections in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_section_wrapper, task): task 
            for task in all_tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            completed += 1
            try:
                ticker, year, section, success, status = future.result()
                if status == "skipped":
                    print(f"⏭️  [{completed}/{len(all_tasks)}] {ticker}/{year}/{section}.json - Skipped (already exists)")
                elif status == "section_not_found":
                    print(f"⏭️  [{completed}/{len(all_tasks)}] {ticker}/{year}/{section}.json - Skipped (section not in filing)")
                elif success:
                    print(f"✅ [{completed}/{len(all_tasks)}] {ticker}/{year}/{section}.json - Uploaded")
                else:
                    print(f"❌ [{completed}/{len(all_tasks)}] {ticker}/{year}/{section}.json - Failed: {status}")
            except Exception as e:
                print(f"❌ [{completed}/{len(all_tasks)}] Exception: {str(e)}")
    
    # Summary
    elapsed_time = time.time() - start_time
    success_rate = (stats["successful"] / stats["total_sections"] * 100) if stats["total_sections"] > 0 else 0
    
    print(f"\n{'=' * 80}")
    print("✅ PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"📊 Total Sections: {stats['total_sections']}")
    print(f"✅ Successful Uploads: {stats['successful']}")
    print(f"⏭️  Skipped (already exists): {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"🔄 API Retries: {stats['api_retries']}")
    print(f"🔄 GCS Retries: {stats['gcs_retries']}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"⏱️  Time Elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"☁️  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 GCS Path: {GCS_BASE_PATH}/<ticker>/<year>/<section>.json")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

