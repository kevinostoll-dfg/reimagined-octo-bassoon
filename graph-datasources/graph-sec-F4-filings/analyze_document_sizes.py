#!/usr/bin/env python3
"""
Analyze document sizes and word counts for Form 4 filings.
Samples across different companies and time periods to get representative statistics.
"""

import os
import sys
import json
import requests
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict

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
        return None

def get_sample_filings(symbol, sample_size=10):
    """Get a sample of filings for a symbol across different time periods."""
    # Get filings from different years to get a good sample
    current_year = datetime.now().year
    samples = []
    
    # Sample from recent years (2023-2025)
    for year in range(current_year - 2, current_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        query = f"issuer.tradingSymbol:{symbol} AND periodOfReport:[{start_date} TO {end_date}]"
        query_params = {
            "query": query,
            "from": 0,
            "size": min(sample_size, 50),
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        result = query_sec_api(query_params)
        if result:
            transactions = result.get("transactions", [])
            samples.extend(transactions[:sample_size // 3])  # Get a few from each year
    
    return samples[:sample_size]

def count_words(text):
    """Count words in text."""
    if not text:
        return 0
    # Remove extra whitespace and split
    words = re.findall(r'\b\w+\b', str(text))
    return len(words)

def extract_all_text(filing):
    """Extract all text content from a filing for word counting."""
    text_parts = []
    
    # Issuer info
    issuer = filing.get("issuer", {})
    if issuer.get("name"):
        text_parts.append(issuer["name"])
    
    # Reporting owner info
    reporting_owner = filing.get("reportingOwner", {})
    if reporting_owner.get("name"):
        text_parts.append(reporting_owner["name"])
    if reporting_owner.get("address"):
        addr = reporting_owner["address"]
        text_parts.extend([v for v in addr.values() if v])
    
    # Relationship info
    relationship = reporting_owner.get("relationship", {})
    if relationship.get("officerTitle"):
        text_parts.append(relationship["officerTitle"])
    if relationship.get("otherText"):
        text_parts.append(relationship["otherText"])
    
    # Non-derivative transactions
    non_deriv = filing.get("nonDerivativeTable", {})
    if non_deriv:
        transactions = non_deriv.get("transactions", [])
        for txn in transactions:
            if txn.get("securityTitle"):
                text_parts.append(txn["securityTitle"])
        
        holdings = non_deriv.get("holdings", [])
        for holding in holdings:
            if holding.get("securityTitle"):
                text_parts.append(holding["securityTitle"])
    
    # Derivative transactions
    deriv = filing.get("derivativeTable", {})
    if deriv:
        transactions = deriv.get("transactions", [])
        for txn in transactions:
            if txn.get("securityTitle"):
                text_parts.append(txn["securityTitle"])
            underlying = txn.get("underlyingSecurity", {})
            if underlying.get("title"):
                text_parts.append(underlying["title"])
        
        holdings = deriv.get("holdings", [])
        for holding in holdings:
            if holding.get("securityTitle"):
                text_parts.append(holding["securityTitle"])
            underlying = holding.get("underlyingSecurity", {})
            if underlying and underlying.get("title"):
                text_parts.append(underlying["title"])
    
    # Footnotes (often contain substantial text)
    footnotes = filing.get("footnotes", [])
    for footnote in footnotes:
        if footnote.get("text"):
            text_parts.append(footnote["text"])
    
    # Remarks
    remarks = filing.get("remarks")
    if remarks:
        text_parts.append(remarks)
    
    # Owner signature
    if filing.get("ownerSignatureName"):
        text_parts.append(filing["ownerSignatureName"])
    
    return " ".join(text_parts)

def analyze_filing(filing):
    """Analyze a single filing for size and word count."""
    # Calculate JSON size
    json_str = json.dumps(filing, ensure_ascii=False)
    json_size_bytes = len(json_str.encode('utf-8'))
    json_size_kb = json_size_bytes / 1024
    
    # Extract and count words
    all_text = extract_all_text(filing)
    word_count = count_words(all_text)
    
    # Count transactions
    non_deriv_txns = len(filing.get("nonDerivativeTable", {}).get("transactions", []))
    deriv_txns = len(filing.get("derivativeTable", {}).get("transactions", []))
    total_txns = non_deriv_txns + deriv_txns
    
    # Count footnotes
    footnote_count = len(filing.get("footnotes", []))
    
    return {
        "json_size_kb": json_size_kb,
        "json_size_bytes": json_size_bytes,
        "word_count": word_count,
        "transactions": total_txns,
        "footnotes": footnote_count,
        "has_remarks": bool(filing.get("remarks")),
        "accession_no": filing.get("accessionNo", "N/A"),
        "filed_at": filing.get("filedAt", "N/A"),
        "period_of_report": filing.get("periodOfReport", "N/A")
    }

def calculate_statistics(analyses):
    """Calculate statistics from a list of analyses."""
    if not analyses:
        return None
    
    stats = {
        "count": len(analyses),
        "json_size": {
            "min_kb": min(a["json_size_kb"] for a in analyses),
            "max_kb": max(a["json_size_kb"] for a in analyses),
            "avg_kb": sum(a["json_size_kb"] for a in analyses) / len(analyses),
            "median_kb": sorted([a["json_size_kb"] for a in analyses])[len(analyses) // 2],
            "total_kb": sum(a["json_size_kb"] for a in analyses)
        },
        "word_count": {
            "min": min(a["word_count"] for a in analyses),
            "max": max(a["word_count"] for a in analyses),
            "avg": sum(a["word_count"] for a in analyses) / len(analyses),
            "median": sorted([a["word_count"] for a in analyses])[len(analyses) // 2],
            "total": sum(a["word_count"] for a in analyses)
        },
        "transactions": {
            "min": min(a["transactions"] for a in analyses),
            "max": max(a["transactions"] for a in analyses),
            "avg": sum(a["transactions"] for a in analyses) / len(analyses),
            "total": sum(a["transactions"] for a in analyses)
        },
        "footnotes": {
            "min": min(a["footnotes"] for a in analyses),
            "max": max(a["footnotes"] for a in analyses),
            "avg": sum(a["footnotes"] for a in analyses) / len(analyses),
            "total": sum(a["footnotes"] for a in analyses)
        }
    }
    
    return stats

def main():
    """Main function."""
    print("="*80)
    print("Form 4 Filing Document Size Analysis")
    print("="*80)
    print(f"Sampling from: {', '.join(TRACKED_SYMBOLS)}")
    print(f"Sample size per symbol: 10 filings")
    print()
    
    all_analyses = []
    symbol_stats = {}
    
    for symbol in TRACKED_SYMBOLS:
        print(f"📊 Sampling {symbol}...")
        filings = get_sample_filings(symbol, sample_size=10)
        
        if not filings:
            print(f"   ⚠️  No filings found for {symbol}")
            continue
        
        print(f"   Found {len(filings)} filings")
        
        analyses = []
        for filing in filings:
            analysis = analyze_filing(filing)
            analyses.append(analysis)
            all_analyses.append(analysis)
        
        stats = calculate_statistics(analyses)
        symbol_stats[symbol] = stats
        
        if stats:
            print(f"   ✅ Analyzed {stats['count']} filings")
            print(f"      Avg size: {stats['json_size']['avg_kb']:.2f} KB")
            print(f"      Avg words: {stats['word_count']['avg']:.0f}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    if all_analyses:
        overall_stats = calculate_statistics(all_analyses)
        
        print(f"\n📄 Document Size (JSON):")
        print(f"   Total samples: {overall_stats['count']}")
        print(f"   Min: {overall_stats['json_size']['min_kb']:.2f} KB")
        print(f"   Max: {overall_stats['json_size']['max_kb']:.2f} KB")
        print(f"   Average: {overall_stats['json_size']['avg_kb']:.2f} KB")
        print(f"   Median: {overall_stats['json_size']['median_kb']:.2f} KB")
        print(f"   Total (all samples): {overall_stats['json_size']['total_kb']:.2f} KB")
        
        print(f"\n📝 Word Count:")
        print(f"   Min: {overall_stats['word_count']['min']:,} words")
        print(f"   Max: {overall_stats['word_count']['max']:,} words")
        print(f"   Average: {overall_stats['word_count']['avg']:.0f} words")
        print(f"   Median: {overall_stats['word_count']['median']:.0f} words")
        print(f"   Total (all samples): {overall_stats['word_count']['total']:,} words")
        
        print(f"\n📊 Transactions per filing:")
        print(f"   Min: {overall_stats['transactions']['min']}")
        print(f"   Max: {overall_stats['transactions']['max']}")
        print(f"   Average: {overall_stats['transactions']['avg']:.1f}")
        print(f"   Total (all samples): {overall_stats['transactions']['total']}")
        
        print(f"\n📌 Footnotes per filing:")
        print(f"   Min: {overall_stats['footnotes']['min']}")
        print(f"   Max: {overall_stats['footnotes']['max']}")
        print(f"   Average: {overall_stats['footnotes']['avg']:.1f}")
        print(f"   Total (all samples): {overall_stats['footnotes']['total']}")
    
    # Per-symbol breakdown
    print("\n" + "="*80)
    print("PER-SYMBOL BREAKDOWN")
    print("="*80)
    print(f"\n{'Symbol':<8} {'Samples':<8} {'Avg KB':<10} {'Avg Words':<12} {'Avg Txns':<10}")
    print("-" * 80)
    
    for symbol in TRACKED_SYMBOLS:
        if symbol in symbol_stats:
            stats = symbol_stats[symbol]
            print(f"{symbol:<8} {stats['count']:<8} {stats['json_size']['avg_kb']:<10.2f} "
                  f"{stats['word_count']['avg']:<12.0f} {stats['transactions']['avg']:<10.1f}")
    
    # Sample details
    print("\n" + "="*80)
    print("SAMPLE FILING DETAILS")
    print("="*80)
    
    # Show a few examples
    print(f"\nSmallest filing:")
    smallest = min(all_analyses, key=lambda x: x["json_size_kb"])
    print(f"   Symbol: {smallest.get('accession_no', 'N/A')[:20]}...")
    print(f"   Size: {smallest['json_size_kb']:.2f} KB")
    print(f"   Words: {smallest['word_count']:,}")
    print(f"   Transactions: {smallest['transactions']}")
    
    print(f"\nLargest filing:")
    largest = max(all_analyses, key=lambda x: x["json_size_kb"])
    print(f"   Symbol: {largest.get('accession_no', 'N/A')[:20]}...")
    print(f"   Size: {largest['json_size_kb']:.2f} KB")
    print(f"   Words: {largest['word_count']:,}")
    print(f"   Transactions: {largest['transactions']}")
    print(f"   Footnotes: {largest['footnotes']}")
    
    print("\n" + "="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
