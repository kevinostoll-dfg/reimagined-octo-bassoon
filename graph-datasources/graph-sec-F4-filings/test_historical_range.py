#!/usr/bin/env python3
"""
Test script to query SEC API for Form 4 filings and verify historical data availability.
This script validates all tracked symbols for 10 years of historical data.
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env')

# Configuration
SEC_API_KEY = os.getenv("SEC_API_KEY")
SEC_API_ENDPOINT = os.getenv("SEC_API_ENDPOINT", "https://api.sec-api.io/insider-trading")
TRACKED_SYMBOLS_STR = os.getenv("TRACKED_SYMBOLS", "TSLA,META,MSFT,AMZN,NFLX,AAPL,NVDA,GOOGL")

if not SEC_API_KEY:
    print("❌ Error: SEC_API_KEY not found in environment variables")
    sys.exit(1)

# Parse tracked symbols
TRACKED_SYMBOLS = [s.strip() for s in TRACKED_SYMBOLS_STR.split(",") if s.strip()]
YEARS_TO_CHECK = 10  # Focus on 10 years as requested

def query_sec_api(query_params):
    """Make a POST request to the SEC API."""
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
        print(f"❌ API request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
        return None

def get_date_range(years_back):
    """Calculate date range for query."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def query_by_date_range(symbol, start_date, end_date, from_offset=0, size=50):
    """Query SEC API for filings within a date range."""
    query = f"issuer.tradingSymbol:{symbol} AND periodOfReport:[{start_date} TO {end_date}]"
    
    query_params = {
        "query": query,
        "from": from_offset,
        "size": size,
        "sort": [{"periodOfReport": {"order": "asc"}}]  # Sort by oldest first
    }
    
    return query_sec_api(query_params)

def find_earliest_filing(symbol, years_back):
    """Find the earliest filing available for a symbol."""
    print(f"\n🔍 Searching for earliest Form 4 filing for {symbol}...")
    print(f"   Checking up to {years_back} years back...")
    
    start_date, end_date = get_date_range(years_back)
    print(f"   Date range: {start_date} to {end_date}")
    
    # Query for the first batch (oldest first)
    result = query_by_date_range(symbol, start_date, end_date, from_offset=0, size=50)
    
    if not result:
        print("❌ Failed to query API")
        return None
    
    total = result.get("total", {})
    total_value = total.get("value", 0)
    total_relation = total.get("relation", "eq")
    
    print(f"\n📊 Query Results:")
    print(f"   Total matching filings: {total_value} ({total_relation})")
    
    transactions = result.get("transactions", [])
    print(f"   Transactions in this batch: {len(transactions)}")
    
    if not transactions:
        print("   ⚠️  No filings found in this date range")
        return None
    
    # Find earliest filing
    earliest = None
    earliest_date = None
    
    for txn in transactions:
        period_of_report = txn.get("periodOfReport")
        if period_of_report:
            if earliest_date is None or period_of_report < earliest_date:
                earliest_date = period_of_report
                earliest = txn
    
    if earliest:
        print(f"\n✅ Earliest filing found:")
        print(f"   Period of Report: {earliest_date}")
        print(f"   Filed At: {earliest.get('filedAt', 'N/A')}")
        print(f"   Accession No: {earliest.get('accessionNo', 'N/A')}")
        
        issuer = earliest.get("issuer", {})
        reporting_owner = earliest.get("reportingOwner", {})
        
        print(f"   Company: {issuer.get('name', 'N/A')} ({issuer.get('tradingSymbol', 'N/A')})")
        print(f"   Insider: {reporting_owner.get('name', 'N/A')}")
        
        # Calculate how many years back
        if earliest_date:
            earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
            years_ago = (datetime.now() - earliest_dt).days / 365.25
            print(f"   Years ago: {years_ago:.2f}")
        
        return earliest
    
    return None

def analyze_historical_coverage(symbol, years_back, verbose=True):
    """Analyze historical data coverage by checking multiple date ranges."""
    if verbose:
        print(f"\n📈 Analyzing historical coverage for {symbol}...")
    
    # Check each year going back
    current_year = datetime.now().year
    results = {}
    
    for year_offset in range(years_back + 1):
        year = current_year - year_offset
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Query for this year
        result = query_by_date_range(symbol, start_date, end_date, from_offset=0, size=1)
        
        if result:
            total = result.get("total", {})
            total_value = total.get("value", 0)
            results[year] = total_value
            
            if verbose:
                status = "✅" if total_value > 0 else "❌"
                print(f"   {status} {year}: {total_value} filings")
        else:
            results[year] = 0
            if verbose:
                print(f"   ❌ {year}: Query failed")
    
    return results

def validate_symbol(symbol, years_back):
    """Validate a single symbol and return summary statistics."""
    start_date, end_date = get_date_range(years_back)
    
    # Query for total count in the 10-year range
    result = query_by_date_range(symbol, start_date, end_date, from_offset=0, size=1)
    
    if not result:
        return {
            "symbol": symbol,
            "status": "error",
            "total_filings": 0,
            "earliest_date": None,
            "years_with_data": 0,
            "oldest_year": None
        }
    
    total = result.get("total", {})
    total_value = total.get("value", 0)
    
    if total_value == 0:
        return {
            "symbol": symbol,
            "status": "no_data",
            "total_filings": 0,
            "earliest_date": None,
            "years_with_data": 0,
            "oldest_year": None
        }
    
    # Get earliest filing
    earliest_result = query_by_date_range(symbol, start_date, end_date, from_offset=0, size=50)
    earliest_date = None
    if earliest_result:
        transactions = earliest_result.get("transactions", [])
        for txn in transactions:
            period = txn.get("periodOfReport")
            if period:
                if earliest_date is None or period < earliest_date:
                    earliest_date = period
    
    # Analyze year-by-year coverage
    coverage = analyze_historical_coverage(symbol, years_back, verbose=False)
    years_with_data = sum(1 for count in coverage.values() if count > 0)
    years_with_data_list = [year for year, count in coverage.items() if count > 0]
    oldest_year = min(years_with_data_list) if years_with_data_list else None
    
    return {
        "symbol": symbol,
        "status": "success",
        "total_filings": total_value,
        "earliest_date": earliest_date,
        "years_with_data": years_with_data,
        "oldest_year": oldest_year,
        "coverage_by_year": coverage
    }

def main():
    """Main function."""
    print("="*80)
    print("SEC API Historical Data Range Validation")
    print("="*80)
    print(f"Tracked Symbols: {', '.join(TRACKED_SYMBOLS)}")
    print(f"API Endpoint: {SEC_API_ENDPOINT}")
    print(f"Years to check: {YEARS_TO_CHECK}")
    print()
    
    # Validate all symbols
    results = []
    for symbol in TRACKED_SYMBOLS:
        print(f"\n{'='*80}")
        print(f"Validating {symbol}...")
        print('='*80)
        
        result = validate_symbol(symbol, YEARS_TO_CHECK)
        results.append(result)
        
        # Print summary for this symbol
        if result["status"] == "error":
            print(f"❌ Error querying {symbol}")
        elif result["status"] == "no_data":
            print(f"⚠️  No filings found for {symbol} in the last {YEARS_TO_CHECK} years")
        else:
            print(f"✅ {symbol}: {result['total_filings']} total filings")
            if result["earliest_date"]:
                earliest_dt = datetime.strptime(result["earliest_date"], "%Y-%m-%d")
                years_ago = (datetime.now() - earliest_dt).days / 365.25
                print(f"   Earliest: {result['earliest_date']} ({years_ago:.1f} years ago)")
            print(f"   Years with data: {result['years_with_data']} out of {YEARS_TO_CHECK + 1}")
            if result["oldest_year"]:
                print(f"   Oldest year: {result['oldest_year']}")
    
    # Overall summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"\n{'Symbol':<8} {'Status':<12} {'Total':<10} {'Earliest':<12} {'Years':<8} {'Oldest':<8}")
    print("-" * 80)
    
    for result in results:
        symbol = result["symbol"]
        status = result["status"]
        total = result["total_filings"]
        earliest = result["earliest_date"] or "N/A"
        years = f"{result['years_with_data']}/{YEARS_TO_CHECK + 1}"
        oldest = str(result["oldest_year"]) if result["oldest_year"] else "N/A"
        
        status_icon = "✅" if status == "success" else "❌" if status == "error" else "⚠️"
        print(f"{symbol:<8} {status_icon} {status:<10} {total:<10} {earliest:<12} {years:<8} {oldest:<8}")
    
    # Statistics
    successful = sum(1 for r in results if r["status"] == "success")
    total_filings_all = sum(r["total_filings"] for r in results)
    
    print("\n" + "-" * 80)
    print(f"Total symbols validated: {len(results)}")
    print(f"Successfully validated: {successful}")
    print(f"Total filings across all symbols: {total_filings_all:,}")
    
    print("\n" + "="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
