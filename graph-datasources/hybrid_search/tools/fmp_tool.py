"""
FMP Proxy Tool for LlamaIndex Agent.

This tool provides access to Financial Modeling Prep (FMP) API through the caching proxy
for financial data including company profiles, financial statements, ratios, news, and market data.
"""

import os
import sys
import json
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config

logger = logging.getLogger(__name__)

# FMP Proxy base URL
FMP_PROXY_URL = "https://api.blacksmith.deerfieldgreen.com/fmp"
REQUEST_TIMEOUT = 30.0


def _make_fmp_request(endpoint_path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Make a request to the FMP proxy API.
    
    Args:
        endpoint_path: FMP API endpoint path (e.g., "stable/profile", "stable/income-statement")
        params: Query parameters for the request
        
    Returns:
        Dictionary containing the API response data or error information
    """
    try:
        url = f"{FMP_PROXY_URL}/{endpoint_path}"
        params = params or {}
        
        logger.info(f"Making FMP proxy request to: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        
        if response.status_code != 200:
            error_msg = f"FMP Proxy API error: {response.status_code} - {response.text[:200]}"
            logger.error(error_msg)
            return {"error": error_msg, "status_code": response.status_code}
        
        data = response.json()
        
        # Handle empty responses
        if not data or (isinstance(data, list) and len(data) == 0):
            return {"data": [], "message": "No data found"}
        
        return {"data": data}
        
    except requests.exceptions.RequestException as e:
        error_msg = f"FMP Proxy request failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error in FMP request: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def get_company_profile(symbol: str) -> str:
    """
    Get company profile information including company description, industry, sector, employees, etc.
    
    WHEN TO USE THIS TOOL:
    - Query asks for company information, profile, description, or basic company details
    - Query asks about industry, sector, employees, or company overview
    - Query asks "What is [company]?" or "Tell me about [company]"
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Financial ratios (use get_financial_ratios instead)
    - Stock news (use get_stock_news instead)
    - Market data/price data (use get_market_data instead)
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT")
        
    Returns:
        Formatted string with company profile information
    """
    try:
        result = _make_fmp_request("stable/profile", params={"symbol": symbol})
        
        if "error" in result:
            return f"Error fetching company profile for {symbol}: {result['error']}"
        
        if "message" in result:
            return f"No company profile found for {symbol}"
        
        profile = result["data"][0] if isinstance(result["data"], list) and len(result["data"]) > 0 else result["data"]
        
        # Format the response
        formatted = f"COMPANY PROFILE FOR {symbol.upper()}:\n\n"
        formatted += f"Company Name: {profile.get('companyName', 'N/A')}\n"
        formatted += f"Exchange: {profile.get('exchangeShortName', 'N/A')}\n"
        formatted += f"Industry: {profile.get('industry', 'N/A')}\n"
        formatted += f"Sector: {profile.get('sector', 'N/A')}\n"
        formatted += f"Description: {profile.get('description', 'N/A')[:500]}...\n" if profile.get('description') else "Description: N/A\n"
        formatted += f"Website: {profile.get('website', 'N/A')}\n"
        formatted += f"CEO: {profile.get('ceo', 'N/A')}\n"
        formatted += f"Employees: {profile.get('fullTimeEmployees', 'N/A')}\n"
        formatted += f"Market Cap: ${profile.get('mktCap', 'N/A')}\n" if profile.get('mktCap') else ""
        formatted += f"Price: ${profile.get('price', 'N/A')}\n" if profile.get('price') else ""
        
        return formatted
        
    except Exception as e:
        error_msg = f"Error processing company profile for {symbol}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def get_financial_statements(symbol: str, statement_type: str = "income-statement", period: str = "quarter", limit: int = 5) -> str:
    """
    Get financial statements (income statement, balance sheet, cash flow) for a company.
    
    WHEN TO USE THIS TOOL:
    - Query asks for income statement, balance sheet, cash flow statement
    - Query asks for revenue, expenses, assets, liabilities, cash flow
    - Query asks for quarterly or annual financial statements
    - Query asks "What are [company]'s financials?" or "Show me [company]'s income statement"
    
    DO NOT use when query asks for:
    - Financial ratios (use get_financial_ratios instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        statement_type: Type of statement - "income-statement", "balance-sheet-statement", or "cash-flow-statement"
        period: "quarter" or "annual"
        limit: Number of periods to return (default: 5)
        
    Returns:
        Formatted string with financial statement data
    """
    try:
        endpoint_map = {
            "income-statement": "stable/income-statement",
            "balance-sheet": "stable/balance-sheet-statement",
            "cash-flow": "stable/cash-flow-statement"
        }
        
        endpoint = endpoint_map.get(statement_type.lower(), "stable/income-statement")
        
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
        
        result = _make_fmp_request(endpoint, params=params)
        
        if "error" in result:
            return f"Error fetching {statement_type} for {symbol}: {result['error']}"
        
        if "message" in result:
            return f"No {statement_type} data found for {symbol}"
        
        statements = result["data"] if isinstance(result["data"], list) else [result["data"]]
        
        # Format the response
        formatted = f"{statement_type.upper().replace('-', ' ')} FOR {symbol.upper()} ({period.upper()}):\n\n"
        
        for i, stmt in enumerate(statements[:limit], 1):
            date = stmt.get("date", "N/A")
            formatted += f"Period {i} - Date: {date}\n"
            
            if statement_type == "income-statement":
                formatted += f"  Revenue: ${stmt.get('revenue', 'N/A')}\n"
                formatted += f"  Cost of Revenue: ${stmt.get('costOfRevenue', 'N/A')}\n"
                formatted += f"  Gross Profit: ${stmt.get('grossProfit', 'N/A')}\n"
                formatted += f"  Operating Expenses: ${stmt.get('operatingExpenses', 'N/A')}\n"
                formatted += f"  Operating Income: ${stmt.get('operatingIncome', 'N/A')}\n"
                formatted += f"  Net Income: ${stmt.get('netIncome', 'N/A')}\n"
                formatted += f"  EPS: ${stmt.get('eps', 'N/A')}\n"
            elif statement_type == "balance-sheet":
                formatted += f"  Total Assets: ${stmt.get('totalAssets', 'N/A')}\n"
                formatted += f"  Total Liabilities: ${stmt.get('totalLiabilities', 'N/A')}\n"
                formatted += f"  Total Equity: ${stmt.get('totalStockholdersEquity', 'N/A')}\n"
                formatted += f"  Cash and Cash Equivalents: ${stmt.get('cashAndCashEquivalents', 'N/A')}\n"
            elif statement_type == "cash-flow":
                formatted += f"  Operating Cash Flow: ${stmt.get('operatingCashFlow', 'N/A')}\n"
                formatted += f"  Investing Cash Flow: ${stmt.get('netCashUsedForInvestingActivites', 'N/A')}\n"
                formatted += f"  Financing Cash Flow: ${stmt.get('netCashUsedProvidedByFinancingActivities', 'N/A')}\n"
                formatted += f"  Free Cash Flow: ${stmt.get('freeCashFlow', 'N/A')}\n"
            
            formatted += "\n"
        
        return formatted
        
    except Exception as e:
        error_msg = f"Error processing financial statements for {symbol}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def get_financial_ratios(symbol: str, period: str = "quarter", limit: int = 5) -> str:
    """
    Get financial ratios and key metrics for a company.
    
    WHEN TO USE THIS TOOL:
    - Query asks for financial ratios, metrics, or KPIs
    - Query asks for P/E ratio, ROE, ROA, debt-to-equity, current ratio, etc.
    - Query asks "What are [company]'s financial ratios?" or "Show me [company]'s key metrics"
    
    DO NOT use when query asks for:
    - Raw financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        period: "quarter" or "annual"
        limit: Number of periods to return (default: 5)
        
    Returns:
        Formatted string with financial ratios and metrics
    """
    try:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
        
        # Get ratios
        ratios_result = _make_fmp_request("stable/ratios", params=params)
        
        # Get key metrics
        metrics_result = _make_fmp_request("stable/key-metrics", params=params)
        
        formatted = f"FINANCIAL RATIOS AND METRICS FOR {symbol.upper()} ({period.upper()}):\n\n"
        
        # Format ratios
        if "error" not in ratios_result and "message" not in ratios_result:
            ratios = ratios_result["data"] if isinstance(ratios_result["data"], list) else [ratios_result["data"]]
            formatted += "FINANCIAL RATIOS:\n"
            for i, ratio in enumerate(ratios[:limit], 1):
                date = ratio.get("date", "N/A")
                formatted += f"\nPeriod {i} - Date: {date}\n"
                formatted += f"  Current Ratio: {ratio.get('currentRatio', 'N/A')}\n"
                formatted += f"  Quick Ratio: {ratio.get('quickRatio', 'N/A')}\n"
                formatted += f"  Debt-to-Equity: {ratio.get('debtEquityRatio', 'N/A')}\n"
                formatted += f"  Return on Equity (ROE): {ratio.get('returnOnEquity', 'N/A')}\n"
                formatted += f"  Return on Assets (ROA): {ratio.get('returnOnAssets', 'N/A')}\n"
                formatted += f"  Profit Margin: {ratio.get('netProfitMargin', 'N/A')}\n"
        
        # Format key metrics
        if "error" not in metrics_result and "message" not in metrics_result:
            metrics = metrics_result["data"] if isinstance(metrics_result["data"], list) else [metrics_result["data"]]
            formatted += "\n\nKEY METRICS:\n"
            for i, metric in enumerate(metrics[:limit], 1):
                date = metric.get("date", "N/A")
                formatted += f"\nPeriod {i} - Date: {date}\n"
                formatted += f"  P/E Ratio: {metric.get('peRatio', 'N/A')}\n"
                formatted += f"  P/B Ratio: {metric.get('pbRatio', 'N/A')}\n"
                formatted += f"  EV/Revenue: {metric.get('evToRevenue', 'N/A')}\n"
                formatted += f"  EV/EBITDA: {metric.get('evToEbitda', 'N/A')}\n"
                formatted += f"  Market Cap: ${metric.get('marketCap', 'N/A')}\n"
                formatted += f"  Enterprise Value: ${metric.get('enterpriseValue', 'N/A')}\n"
        
        return formatted
        
    except Exception as e:
        error_msg = f"Error processing financial ratios for {symbol}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def get_stock_news(symbol: str, limit: int = 10) -> str:
    """
    Get recent stock news and articles for a company.
    
    WHEN TO USE THIS TOOL:
    - Query asks for news, articles, press releases, or recent updates about a company
    - Query asks "What's the latest news about [company]?" or "Show me recent articles about [company]"
    - Query asks for press releases or company announcements
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        limit: Number of news articles to return (default: 10)
        
    Returns:
        Formatted string with stock news articles
    """
    try:
        # Calculate date range (last 90 days)
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        params = {
            "symbols": symbol,
            "from": from_date,
            "to": to_date,
            "limit": limit,
            "page": 0
        }
        
        result = _make_fmp_request("stable/news/stock", params=params)
        
        if "error" in result:
            return f"Error fetching stock news for {symbol}: {result['error']}"
        
        if "message" in result:
            return f"No stock news found for {symbol}"
        
        articles = result["data"] if isinstance(result["data"], list) else [result["data"]]
        
        formatted = f"STOCK NEWS FOR {symbol.upper()} (Last 90 days):\n\n"
        
        for i, article in enumerate(articles[:limit], 1):
            formatted += f"{i}. {article.get('title', 'N/A')}\n"
            formatted += f"   Published: {article.get('publishedDate', article.get('date', 'N/A'))}\n"
            formatted += f"   Source: {article.get('site', article.get('publisher', 'N/A'))}\n"
            formatted += f"   URL: {article.get('url', article.get('link', 'N/A'))}\n"
            if article.get('text'):
                snippet = article['text'][:200] + "..." if len(article['text']) > 200 else article['text']
                formatted += f"   Snippet: {snippet}\n"
            formatted += "\n"
        
        return formatted
        
    except Exception as e:
        error_msg = f"Error processing stock news for {symbol}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def get_analyst_estimates(symbol: str, period: str = "annual", limit: int = 5) -> str:
    """
    Get analyst estimates and price targets for a company.
    
    WHEN TO USE THIS TOOL:
    - Query asks for analyst estimates, price targets, earnings estimates, or analyst ratings
    - Query asks "What do analysts predict for [company]?" or "Show me analyst estimates for [company]"
    - Query asks for price targets or consensus estimates
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        period: "quarter" or "annual"
        limit: Number of periods to return (default: 5)
        
    Returns:
        Formatted string with analyst estimates and price targets
    """
    try:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
        
        # Get analyst estimates
        estimates_result = _make_fmp_request("stable/analyst-estimates", params=params)
        
        # Get price target consensus
        price_target_result = _make_fmp_request("stable/price-target-consensus", params={"symbol": symbol})
        
        formatted = f"ANALYST ESTIMATES FOR {symbol.upper()} ({period.upper()}):\n\n"
        
        # Format analyst estimates
        if "error" not in estimates_result and "message" not in estimates_result:
            estimates = estimates_result["data"] if isinstance(estimates_result["data"], list) else [estimates_result["data"]]
            formatted += "ANALYST ESTIMATES:\n"
            for i, est in enumerate(estimates[:limit], 1):
                date = est.get("date", "N/A")
                formatted += f"\nPeriod {i} - Date: {date}\n"
                formatted += f"  Revenue Estimate: ${est.get('estimatedRevenueAvg', 'N/A')}\n"
                formatted += f"  Revenue High: ${est.get('estimatedRevenueHigh', 'N/A')}\n"
                formatted += f"  Revenue Low: ${est.get('estimatedRevenueLow', 'N/A')}\n"
                formatted += f"  EPS Estimate: ${est.get('estimatedEpsAvg', 'N/A')}\n"
                formatted += f"  EPS High: ${est.get('estimatedEpsHigh', 'N/A')}\n"
                formatted += f"  EPS Low: ${est.get('estimatedEpsLow', 'N/A')}\n"
        
        # Format price target consensus
        if "error" not in price_target_result and "message" not in price_target_result:
            pt = price_target_result["data"]
            if isinstance(pt, list) and len(pt) > 0:
                pt = pt[0]
            formatted += "\n\nPRICE TARGET CONSENSUS:\n"
            formatted += f"  Consensus Price Target: ${pt.get('priceTargetConsensus', 'N/A')}\n"
            formatted += f"  Number of Analysts: {pt.get('numberOfAnalysts', 'N/A')}\n"
            formatted += f"  Price Target High: ${pt.get('priceTargetHigh', 'N/A')}\n"
            formatted += f"  Price Target Low: ${pt.get('priceTargetLow', 'N/A')}\n"
        
        return formatted
        
    except Exception as e:
        error_msg = f"Error processing analyst estimates for {symbol}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def get_market_data(symbol: str, interval: str = "1day", limit: int = 30) -> str:
    """
    Get historical market data (price, volume) for a stock.
    
    WHEN TO USE THIS TOOL:
    - Query asks for stock price, market data, historical prices, or trading volume
    - Query asks "What is [company]'s stock price?" or "Show me [company]'s price history"
    - Query asks for price charts, volume data, or market performance
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Financial ratios (use get_financial_ratios instead)
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        interval: Time interval - "1day", "5day", "1month", "3month", "1year", "5year"
        limit: Number of data points to return (default: 30)
        
    Returns:
        Formatted string with market data
    """
    try:
        # Map interval to endpoint
        interval_map = {
            "1day": "1day",
            "5day": "5day",
            "1month": "1month",
            "3month": "3month",
            "1year": "1year",
            "5year": "5year"
        }
        
        interval_path = interval_map.get(interval.lower(), "1day")
        endpoint = f"stable/historical-price-eod/light"
        
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        result = _make_fmp_request(endpoint, params=params)
        
        if "error" in result:
            return f"Error fetching market data for {symbol}: {result['error']}"
        
        if "message" in result:
            return f"No market data found for {symbol}"
        
        data_points = result["data"] if isinstance(result["data"], list) else [result["data"]]
        
        formatted = f"HISTORICAL MARKET DATA FOR {symbol.upper()} (Last {limit} days):\n\n"
        
        for i, point in enumerate(data_points[:limit], 1):
            date = point.get("date", "N/A")
            formatted += f"{i}. Date: {date}\n"
            formatted += f"   Open: ${point.get('open', 'N/A')}\n"
            formatted += f"   High: ${point.get('high', 'N/A')}\n"
            formatted += f"   Low: ${point.get('low', 'N/A')}\n"
            formatted += f"   Close: ${point.get('close', 'N/A')}\n"
            formatted += f"   Volume: {point.get('volume', 'N/A')}\n"
            formatted += f"   Change: {point.get('change', 'N/A')}\n"
            formatted += f"   Change %: {point.get('changePercent', 'N/A')}\n"
            formatted += "\n"
        
        return formatted
        
    except Exception as e:
        error_msg = f"Error processing market data for {symbol}: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Create FunctionTool instances for the agent
from llama_index.core.tools import FunctionTool

fmp_company_profile_tool = FunctionTool.from_defaults(
    fn=get_company_profile,
    name="get_company_profile",
    description="""Get company profile information including company description, industry, sector, employees, CEO, website, and market cap.
    
    WHEN TO USE THIS TOOL:
    - Query asks for company information, profile, description, or basic company details
    - Query asks about industry, sector, employees, or company overview
    - Query asks "What is [company]?" or "Tell me about [company]"
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Financial ratios (use get_financial_ratios instead)
    - Stock news (use get_stock_news instead)
    - Market data/price data (use get_market_data instead)
    
    Example queries:
    - "What is Tesla's company profile?"
    - "Tell me about Apple's industry and sector"
    - "What does Microsoft do?"
    """
)

fmp_financial_statements_tool = FunctionTool.from_defaults(
    fn=get_financial_statements,
    name="get_financial_statements",
    description="""Get financial statements (income statement, balance sheet, cash flow) for a company.
    
    WHEN TO USE THIS TOOL:
    - Query asks for income statement, balance sheet, cash flow statement
    - Query asks for revenue, expenses, assets, liabilities, cash flow
    - Query asks for quarterly or annual financial statements
    - Query asks "What are [company]'s financials?" or "Show me [company]'s income statement"
    
    DO NOT use when query asks for:
    - Financial ratios (use get_financial_ratios instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Example queries:
    - "What is Tesla's revenue?"
    - "Show me Apple's income statement"
    - "What are Microsoft's quarterly financials?"
    """
)

fmp_financial_ratios_tool = FunctionTool.from_defaults(
    fn=get_financial_ratios,
    name="get_financial_ratios",
    description="""Get financial ratios and key metrics for a company (P/E ratio, ROE, ROA, debt-to-equity, etc.).
    
    WHEN TO USE THIS TOOL:
    - Query asks for financial ratios, metrics, or KPIs
    - Query asks for P/E ratio, ROE, ROA, debt-to-equity, current ratio, etc.
    - Query asks "What are [company]'s financial ratios?" or "Show me [company]'s key metrics"
    
    DO NOT use when query asks for:
    - Raw financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Example queries:
    - "What is Tesla's P/E ratio?"
    - "Show me Apple's financial ratios"
    - "What are Microsoft's key metrics?"
    """
)

fmp_stock_news_tool = FunctionTool.from_defaults(
    fn=get_stock_news,
    name="get_stock_news",
    description="""Get recent stock news and articles for a company.
    
    WHEN TO USE THIS TOOL:
    - Query asks for news, articles, press releases, or recent updates about a company
    - Query asks "What's the latest news about [company]?" or "Show me recent articles about [company]"
    - Query asks for press releases or company announcements
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Example queries:
    - "What's the latest news about Tesla?"
    - "Show me recent articles about Apple"
    - "What press releases has Microsoft issued?"
    """
)

fmp_analyst_estimates_tool = FunctionTool.from_defaults(
    fn=get_analyst_estimates,
    name="get_analyst_estimates",
    description="""Get analyst estimates and price targets for a company.
    
    WHEN TO USE THIS TOOL:
    - Query asks for analyst estimates, price targets, earnings estimates, or analyst ratings
    - Query asks "What do analysts predict for [company]?" or "Show me analyst estimates for [company]"
    - Query asks for price targets or consensus estimates
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Market data (use get_market_data instead)
    
    Example queries:
    - "What do analysts predict for Tesla?"
    - "Show me analyst estimates for Apple"
    - "What is Microsoft's price target?"
    """
)

fmp_market_data_tool = FunctionTool.from_defaults(
    fn=get_market_data,
    name="get_market_data",
    description="""Get historical market data (price, volume) for a stock.
    
    WHEN TO USE THIS TOOL:
    - Query asks for stock price, market data, historical prices, or trading volume
    - Query asks "What is [company]'s stock price?" or "Show me [company]'s price history"
    - Query asks for price charts, volume data, or market performance
    
    DO NOT use when query asks for:
    - Financial statements (use get_financial_statements instead)
    - Company profile (use get_company_profile instead)
    - Financial ratios (use get_financial_ratios instead)
    
    Example queries:
    - "What is Tesla's stock price?"
    - "Show me Apple's price history"
    - "What is Microsoft's trading volume?"
    """
)

if __name__ == "__main__":
    # Test the tools
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing FMP Tools ===\n")
    
    symbol = "AAPL"
    
    print(f"1. Company Profile for {symbol}:")
    print(get_company_profile(symbol))
    print("\n" + "="*80 + "\n")
    
    print(f"2. Financial Statements for {symbol}:")
    print(get_financial_statements(symbol, statement_type="income-statement", limit=2))
    print("\n" + "="*80 + "\n")
    
    print(f"3. Financial Ratios for {symbol}:")
    print(get_financial_ratios(symbol, limit=2))
    print("\n" + "="*80 + "\n")
    
    print(f"4. Stock News for {symbol}:")
    print(get_stock_news(symbol, limit=3))
    print("\n" + "="*80 + "\n")
    
    print(f"5. Analyst Estimates for {symbol}:")
    print(get_analyst_estimates(symbol, limit=2))
    print("\n" + "="*80 + "\n")
    
    print(f"6. Market Data for {symbol}:")
    print(get_market_data(symbol, limit=5))
    print("\n=== Test Complete ===")

