"""
Example: Using Weekly Market Analysis with Agent

This demonstrates how to use the agent for weekly market analysis.
The agent will automatically use datetime tools for temporal awareness
and filter queries appropriately.
"""

import sys
import json
from typing import Dict, Any, Optional

try:
    from graph_agent import get_agent, run_agent_query
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install missing dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def run_weekly_analysis_for_symbol(symbol: str):
    """
    Run weekly market analysis report for a given symbol using the agent.
    
    This generates a comprehensive 4,000-5,000 word report (not JSON) using
    the weekly_market_analysis_tool.
    
    The agent will automatically:
    - Get current date/week range using datetime tools
    - Query Milvus with date filters
    - Query Graph database with date ranges
    - Generate comprehensive weekly market report
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
    """
    print(f"\n{'='*70}")
    print(f"Weekly Market Analysis Report for {symbol}")
    print(f"{'='*70}\n")
    
    # Use agent to generate comprehensive weekly market report
    # The agent will use weekly_market_analysis_tool which generates full reports
    query = f"""Use the weekly_market_analysis tool to generate a comprehensive weekly US equity market analysis report.
    
    Focus on {symbol} as the primary stock of interest. The report should:
    - Be 4,000-5,000 words
    - Cover the current trading week
    - Include comprehensive market analysis with {symbol} as a key focus
    - Follow the professional report structure with all sections
    
    The agent should first get the current trading week dates, then query data sources
    with appropriate date filters, and generate the comprehensive report."""
    
    print("Querying agent for comprehensive weekly market report...")
    print(f"Focus: {symbol}\n")
    
    response = run_agent_query(query)
    
    print("\n" + "="*70)
    print("WEEKLY MARKET ANALYSIS REPORT")
    print("="*70)
    print(response)
    
    print("\n" + "="*70)
    print("REPORT COMPLETE")
    print("="*70)
    
    return {
        "symbol": symbol,
        "report": response
    }


def run_weekly_analysis_for_all_symbols():
    """Run weekly analysis for all symbols using the agent"""
    from prompts import load_symbols
    
    symbols = load_symbols()
    results = {}
    
    print(f"\nRunning weekly analysis for {len(symbols)} symbols using agent...\n")
    
    for symbol in symbols:
        try:
            result = run_weekly_analysis_for_symbol(symbol)
            results[symbol] = result
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run weekly market analysis report (defaults to TSLA)")
    parser.add_argument("symbol", nargs="?", default="TSLA", help="Stock ticker symbol (default: TSLA)")
    parser.add_argument("--output", "-o", help="Output file path for report")
    
    args = parser.parse_args()
    
    # Always use TSLA as default (already set in argparse, but ensure uppercase)
    symbol = args.symbol.upper() if args.symbol else "TSLA"
    
    # Run weekly market analysis report
    result = run_weekly_analysis_for_symbol(symbol)
    
    # Save to file if requested
    if args.output and result:
        with open(args.output, 'w') as f:
            f.write(result.get("report", ""))
        print(f"\n✅ Saved report to {args.output}")
