"""
Script to check available TSLA data in Milvus and Memgraph databases.

This script queries both databases to see what data exists for Tesla (TSLA)
and what date ranges are available.
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from tools.milvus_search_tool import milvus_hybrid_search_func
    from tools.graphrag_tool import graph_rag_func
    from prompts import get_current_trading_week, format_date_for_prompt
    from tools.datetime_tool import get_current_datetime_info, get_trading_week_range
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install missing dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def extract_dates_from_milvus_results(results: str) -> List[str]:
    """Extract date information from Milvus results."""
    dates = []
    # Look for date patterns in the results
    import re
    # Pattern for dates like "2026-01-05", "Jan 05, 2026", etc.
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # ISO format
        r'[A-Z][a-z]{2} \d{1,2}, \d{4}',  # "Jan 05, 2026"
        r'metadata\["date"\]\s*==\s*"([^"]+)"',  # metadata["date"] == "..."
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, results)
        dates.extend(matches)
    
    return list(set(dates))  # Remove duplicates


def check_milvus_data(symbol: str = "TSLA", limit: int = 20) -> Dict[str, Any]:
    """Check what TSLA data exists in Milvus."""
    print(f"\n{'='*70}")
    print(f"CHECKING MILVUS DATA FOR {symbol}")
    print(f"{'='*70}\n")
    
    results = {
        "symbol": symbol,
        "total_results": 0,
        "dates_found": [],
        "sample_results": [],
        "error": None
    }
    
    try:
        # Query without date filter first to see all available data
        print("Query 1: Searching Milvus for ALL TSLA data (no date filter)...")
        all_results = milvus_hybrid_search_func(
            query=f"Weekly market analysis for {symbol}",
            limit=limit,
            filter_expr=f'metadata["ticker"] == "{symbol}"'
        )
        
        print(f"Results: {all_results[:500]}...\n")
        
        # Extract dates from results
        dates = extract_dates_from_milvus_results(all_results)
        results["dates_found"] = dates
        results["sample_results"] = all_results[:1000]  # First 1000 chars
        
        # Count results (rough estimate)
        if "No results found" in all_results:
            results["total_results"] = 0
        else:
            # Try to count result entries
            result_count = all_results.count("Document") + all_results.count("Result")
            results["total_results"] = max(1, result_count) if result_count > 0 else 1
        
        # Query with current week date filter
        week_start, week_end, trading_days = get_current_trading_week()
        week_start_iso = week_start.strftime("%Y-%m-%d")
        week_end_iso = week_end.strftime("%Y-%m-%d")
        
        print(f"\nQuery 2: Searching Milvus for TSLA data from THIS WEEK ({week_start_iso} to {week_end_iso})...")
        week_filter = f'metadata["ticker"] == "{symbol}" and metadata["date"] >= "{week_start_iso}" and metadata["date"] <= "{week_end_iso}"'
        
        week_results = milvus_hybrid_search_func(
            query=f"Weekly market analysis for {symbol}",
            limit=limit,
            filter_expr=week_filter
        )
        
        print(f"Week Results: {week_results[:500]}...\n")
        
        results["this_week_results"] = week_results[:1000]
        results["this_week_filter"] = week_filter
        
        # Try alternative date field names
        print("\nQuery 3: Trying alternative date field names...")
        alt_filters = [
            f'metadata["ticker"] == "{symbol}" and metadata["created_at"] >= "{week_start_iso}"',
            f'metadata["ticker"] == "{symbol}" and metadata["timestamp"] >= "{week_start_iso}"',
            f'metadata["ticker"] == "{symbol}" and metadata["report_date"] >= "{week_start_iso}"',
        ]
        
        for alt_filter in alt_filters:
            try:
                alt_results = milvus_hybrid_search_func(
                    query=f"{symbol} data",
                    limit=5,
                    filter_expr=alt_filter
                )
                if "No results found" not in alt_results:
                    print(f"  ✓ Found results with filter: {alt_filter[:80]}...")
                    results["alternative_filters"] = results.get("alternative_filters", [])
                    results["alternative_filters"].append({
                        "filter": alt_filter,
                        "results": alt_results[:500]
                    })
            except Exception as e:
                print(f"  ✗ Filter failed: {alt_filter[:80]}... - {e}")
        
    except Exception as e:
        print(f"❌ Error querying Milvus: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
    
    return results


def check_memgraph_data(symbol: str = "TSLA") -> Dict[str, Any]:
    """Check what TSLA data exists in Memgraph."""
    print(f"\n{'='*70}")
    print(f"CHECKING MEMGRAPH DATA FOR {symbol}")
    print(f"{'='*70}\n")
    
    results = {
        "symbol": symbol,
        "queries_run": [],
        "results": [],
        "error": None
    }
    
    try:
        # Query 1: General TSLA information
        print("Query 1: General information about TSLA...")
        query1 = f"What information do we have about {symbol}?"
        result1 = graph_rag_func(query1)
        results["queries_run"].append(query1)
        results["results"].append({
            "query": query1,
            "result": result1[:1000] if result1 else "No results"
        })
        print(f"Result: {result1[:500] if result1 else 'No results'}...\n")
        
        # Query 2: TSLA with date context
        week_start, week_end, trading_days = get_current_trading_week()
        week_start_str = format_date_for_prompt(week_start)
        week_end_str = format_date_for_prompt(week_end)
        
        print(f"Query 2: TSLA information from {week_start_str} to {week_end_str}...")
        query2 = f"What information do we have about {symbol} from {week_start_str} to {week_end_str}? Include earnings, news, and analyst opinions from this week."
        result2 = graph_rag_func(query2)
        results["queries_run"].append(query2)
        results["results"].append({
            "query": query2,
            "result": result2[:1000] if result2 else "No results"
        })
        print(f"Result: {result2[:500] if result2 else 'No results'}...\n")
        
        # Query 3: TSLA earnings and transcripts
        print("Query 3: TSLA earnings and transcripts...")
        query3 = f"What earnings transcripts or earnings information do we have about {symbol}?"
        result3 = graph_rag_func(query3)
        results["queries_run"].append(query3)
        results["results"].append({
            "query": query3,
            "result": result3[:1000] if result3 else "No results"
        })
        print(f"Result: {result3[:500] if result3 else 'No results'}...\n")
        
        # Query 4: TSLA news and statements
        print("Query 4: TSLA news and statements...")
        query4 = f"What statements, news, or mentions do we have about {symbol}?"
        result4 = graph_rag_func(query4)
        results["queries_run"].append(query4)
        results["results"].append({
            "query": query4,
            "result": result4[:1000] if result4 else "No results"
        })
        print(f"Result: {result4[:500] if result4 else 'No results'}...\n")
        
    except Exception as e:
        print(f"❌ Error querying Memgraph: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
    
    return results


def main():
    """Main function to check TSLA data in both databases."""
    symbol = "TSLA"
    
    print("\n" + "="*70)
    print("TSLA DATA AVAILABILITY CHECK")
    print("="*70)
    
    # Get current temporal context
    print("\n" + "="*70)
    print("CURRENT TEMPORAL CONTEXT")
    print("="*70)
    datetime_info = get_current_datetime_info()
    print(datetime_info)
    
    week_range = get_trading_week_range()
    print(week_range)
    
    # Check Milvus
    milvus_results = check_milvus_data(symbol)
    
    # Check Memgraph
    memgraph_results = check_memgraph_data(symbol)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n📊 MILVUS ({symbol}):")
    print(f"  - Total results found: {milvus_results.get('total_results', 0)}")
    print(f"  - Dates found: {len(milvus_results.get('dates_found', []))}")
    if milvus_results.get('dates_found'):
        print(f"  - Sample dates: {milvus_results['dates_found'][:5]}")
    if milvus_results.get('error'):
        print(f"  - Error: {milvus_results['error']}")
    
    print(f"\n🕸️  MEMGRAPH ({symbol}):")
    print(f"  - Queries run: {len(memgraph_results.get('queries_run', []))}")
    print(f"  - Results found: {len([r for r in memgraph_results.get('results', []) if r.get('result') and 'No results' not in r.get('result', '')])}")
    if memgraph_results.get('error'):
        print(f"  - Error: {memgraph_results['error']}")
    
    # Save results to JSON file
    output_file = f"tsla_data_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "symbol": symbol,
        "check_date": datetime.now().isoformat(),
        "temporal_context": {
            "datetime_info": datetime_info,
            "week_range": week_range
        },
        "milvus": milvus_results,
        "memgraph": memgraph_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to: {output_file}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

