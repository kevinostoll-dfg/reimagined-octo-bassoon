"""
Prompts for Hybrid Search Queries
Contains reusable prompt templates for various analysis tasks
"""

from typing import Dict, List, Tuple
import json
from pathlib import Path
from datetime import datetime, timedelta

# Load symbols from symbols.json
def load_symbols() -> List[str]:
    """Load symbols from symbols.json file"""
    symbols_path = Path(__file__).parent / "symbols.json"
    try:
        with open(symbols_path, 'r') as f:
            symbols = json.load(f)
        return symbols if isinstance(symbols, list) else []
    except Exception as e:
        print(f"Warning: Could not load symbols.json: {e}")
        return []


def get_current_trading_week() -> Tuple[datetime, datetime, List[datetime]]:
    """
    Calculate the current trading week (Monday to Friday).
    
    Returns:
        Tuple of (week_start, week_end, trading_days)
        - week_start: Monday of current week
        - week_end: Friday of current week (or today if before Friday)
        - trading_days: List of all trading days (Mon-Fri) in the week
    """
    today = datetime.now()
    
    # Find Monday of current week (weekday: 0=Monday, 6=Sunday)
    days_since_monday = today.weekday()
    week_start = today - timedelta(days=days_since_monday)
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Find Friday of current week (or today if it's before Friday)
    days_until_friday = 4 - days_since_monday  # 4 = Friday (0-indexed)
    if days_until_friday < 0:
        # If it's Saturday/Sunday, use Friday of current week
        week_end = week_start + timedelta(days=4)
    else:
        # Use today if it's Mon-Fri, otherwise use Friday
        week_end = min(today, week_start + timedelta(days=4))
    
    week_end = week_end.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Generate list of trading days (Mon-Fri)
    trading_days = []
    for i in range(5):  # Monday (0) to Friday (4)
        day = week_start + timedelta(days=i)
        if day <= today:  # Only include days up to today
            trading_days.append(day)
    
    return week_start, week_end, trading_days


def format_date_for_prompt(dt: datetime) -> str:
    """Format datetime as 'MMM DD, YYYY' (e.g., 'Oct 14, 2024')"""
    return dt.strftime("%b %d, %Y")


def format_day_abbreviation(dt: datetime) -> str:
    """Format datetime as day abbreviation (e.g., 'Mon', 'Tue')"""
    return dt.strftime("%a")


def get_weekly_market_analysis_prompt(symbol: str) -> str:
    """
    Generate a comprehensive weekly market analysis prompt for a given symbol.
    
    This prompt is designed to be used with hybrid search to gather:
    - Recent market performance and price movements
    - News and sentiment analysis
    - Earnings and financial updates
    - Analyst opinions and recommendations
    - Technical indicators and trends
    - Sector and market comparisons
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
    
    Returns:
        Formatted prompt string for hybrid search
    """
    prompt = f"""Provide a comprehensive weekly market analysis for {symbol}. 

Please gather and analyze the following information:

1. **Price Performance & Trends**
   - Current price and weekly price movement
   - High and low points during the week
   - Volume trends and trading activity
   - Technical indicators and chart patterns
   - Support and resistance levels

2. **News & Sentiment Analysis**
   - Major news events affecting {symbol} this week
   - Market sentiment (bullish/bearish/neutral)
   - Key announcements, earnings reports, or corporate actions
   - Regulatory or industry developments
   - Analyst upgrades/downgrades

3. **Financial Metrics & Fundamentals**
   - Recent earnings data and guidance
   - Key financial ratios (P/E, P/B, etc.)
   - Revenue and profit trends
   - Balance sheet highlights
   - Cash flow analysis

4. **Market Context & Comparisons**
   - Performance vs. sector peers
   - Performance vs. market indices (S&P 500, NASDAQ, etc.)
   - Sector rotation and industry trends
   - Macroeconomic factors affecting the stock

5. **Key Insights & Outlook**
   - Summary of the most important developments
   - Risk factors and concerns
   - Growth opportunities and catalysts
   - Short-term and medium-term outlook
   - Investment thesis and recommendations

Use both graph_search and milvus_search tools to gather comprehensive information from:
- Earnings transcripts and financial statements
- News articles and market reports
- Analyst research and recommendations
- Historical price and volume data
- Company statements and announcements

Provide a structured analysis with clear sections and actionable insights."""
    
    return prompt


def get_symbol_analysis_prompts() -> Dict[str, str]:
    """
    Generate weekly market analysis prompts for all symbols in symbols.json
    
    Returns:
        Dictionary mapping symbol to its analysis prompt
    """
    symbols = load_symbols()
    prompts = {}
    
    for symbol in symbols:
        prompts[symbol] = get_weekly_market_analysis_prompt(symbol)
    
    return prompts


def get_structured_weekly_analysis_prompt(symbol: str, milvus_results: str, graph_results: str, 
                                         current_date: datetime = None) -> str:
    """
    Generate a prompt that requests structured JSON output matching WeeklyWrapUpData schema.
    
    This prompt takes the raw search results and asks the LLM to structure them into
    the WeeklyWrapUpData format.
    
    Args:
        symbol: Stock ticker symbol (e.g., "TSLA")
        milvus_results: Raw text results from Milvus hybrid search
        graph_results: Raw text results from graph database search
        current_date: Current date (defaults to datetime.now() if not provided)
    
    Returns:
        Formatted prompt string requesting structured JSON output
    """
    # Calculate temporal context
    if current_date is None:
        current_date = datetime.now()
    
    week_start, week_end, trading_days = get_current_trading_week()
    week_start_str = format_date_for_prompt(week_start)
    week_end_str = format_date_for_prompt(week_end)
    current_date_str = format_date_for_prompt(current_date)
    
    # Format trading days for reference
    trading_days_info = []
    for day in trading_days:
        trading_days_info.append({
            "day": format_day_abbreviation(day),
            "date": format_date_for_prompt(day)
        })
    
    prompt = f"""You are a financial analyst generating a comprehensive weekly market analysis report for {symbol}.

=== TEMPORAL CONTEXT (CRITICAL - USE THESE EXACT DATES) ===
Current Date: {current_date_str}
Analysis Week Range: {week_start_str} to {week_end_str}
Trading Days This Week:
{chr(10).join([f"  - {day['day']}, {day['date']}" for day in trading_days_info])}

IMPORTANT: You MUST use the exact dates provided above. Do NOT use placeholder dates or dates from the data sources unless they match the week range above.

Based on the following data gathered from multiple sources, generate a structured JSON response matching the WeeklyWrapUpData schema.

=== DATA FROM VECTOR DATABASE (Milvus) ===
{milvus_results}

=== DATA FROM GRAPH DATABASE (Memgraph) ===
{graph_results}

=== TASK ===
Analyze the above data and generate a complete weekly market analysis in JSON format matching this exact structure:

{{
  "weekRange": {{
    "start": "Oct 14, 2024",  // Start date of the week
    "end": "Oct 21, 2024"      // End date of the week
  }},
  "performance": {{
    "priceChange": 0.0,         // Price change in dollars
    "priceChangePercent": 0.0,   // Price change percentage
    "volume": "1.2B",            // Volume as formatted string
    "volumeChange": 0.0,         // Volume change percentage
    "high": 0.0,                 // Weekly high price
    "low": 0.0                   // Weekly low price
  }},
  "keyHighlights": [
    {{
      "type": "positive",        // "positive" | "negative" | "neutral"
      "category": "Earnings",    // Category name
      "title": "Title",          // Highlight title
      "description": "Description", // Full description
      "impact": "high",          // "high" | "medium" | "low"
      "date": "Oct 15, 2024"    // Date string
    }}
  ],
  "dailyPerformance": [
    {{
      "day": "Mon",              // Day abbreviation
      "date": "Oct 14, 2024",   // Date string
      "price": 0.0,             // Closing price
      "change": 0.0,            // Price change in dollars
      "changePercent": 0.0,     // Price change percentage
      "volume": 0               // Trading volume
    }}
  ],
  "sectorComparison": {{
    "sectorPerformance": 0.0,   // Sector performance percentage
    "sectorAverage": 0.0,       // Sector average performance
    "rank": 1,                  // Rank within sector (1 = best)
    "totalInSector": 0          // Total companies in sector
  }},
  "newsSummary": {{
    "totalNews": 0,             // Total news articles
    "bullish": 0,               // Number of bullish articles
    "bearish": 0,               // Number of bearish articles
    "neutral": 0,               // Number of neutral articles
    "topStory": {{
      "title": "Story Title",
      "source": "Source Name",
      "sentiment": "bullish",   // "bullish" | "bearish" | "neutral"
      "impact": "High impact description"
    }}
  }},
  "analystActivity": {{
    "upgrades": 0,              // Number of upgrades
    "downgrades": 0,            // Number of downgrades
    "initiations": 0,          // Number of initiations
    "priceTargetChanges": 0,   // Number of price target changes
    "averagePriceTarget": 0.0, // Average price target
    "priceTargetChange": 0.0   // Price target change percentage
  }},
  "optionsActivity": {{
    "unusualVolume": 0,         // Unusual volume count
    "callPutRatio": 0.0,        // Call/Put ratio
    "largestTrade": {{          // Can be null if no options data available
      "type": "CALL",           // MUST be "CALL" or "PUT" (never "N/A" or other values)
      "strike": 0.0,            // Strike price
      "volume": 0,              // Trade volume
      "premium": "$1.2M"        // Premium as formatted string
    }}
    // OR set to null if no options data: "largestTrade": null
  }},
  "institutionalActivity": {{
    "netFlow": 0.0,             // Net flow in dollars
    "netFlowPercent": 0.0,      // Net flow percentage
    "topBuyer": {{
      "name": "Institution Name",
      "amount": "$100M"
    }},
    "topSeller": {{
      "name": "Institution Name",
      "amount": "$50M"
    }}
  }},
  "swot": {{
    "strengths": ["Strength 1", "Strength 2"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "opportunities": ["Opportunity 1", "Opportunity 2"],
    "threats": ["Threat 1", "Threat 2"]
  }}
}}

=== INSTRUCTIONS ===
1. **TEMPORAL ACCURACY (CRITICAL)**: 
   - Use weekRange.start = "{week_start_str}" and weekRange.end = "{week_end_str}" EXACTLY as provided
   - For dailyPerformance, use ONLY the trading days listed above ({len(trading_days_info)} days)
   - For keyHighlights dates, use dates from the week range ({week_start_str} to {week_end_str})
   - If data sources mention dates outside this range, prioritize data from WITHIN this week
   
2. Extract all relevant information from the provided data sources that occurred during the week ({week_start_str} to {week_end_str})

3. Calculate metrics based on the data (price changes, volumes, percentages) for the specified week

4. Identify key highlights from news, earnings, and analyst reports that occurred during this week

5. Generate daily performance breakdown for each trading day listed above (use exact dates provided)

6. Compare performance to sector peers if data is available

7. Analyze news sentiment and categorize articles as bullish/bearish/neutral (focus on news from this week)

8. Summarize analyst activity (upgrades, downgrades, price targets) that occurred during this week

9. Include options activity if available in the data for this week
   - If no options data is available, set "largestTrade" to null (not "N/A")
   - If options data exists, "largestTrade.type" MUST be either "CALL" or "PUT" (never "N/A" or other values)
   - Use default values (type="CALL", strike=0.0, volume=0, premium="$0") if partial data is available

10. Include institutional activity if available in the data for this week

11. Perform SWOT analysis based on gathered information

12. If specific data is not available, use reasonable defaults (0 for numbers, empty arrays for lists)

13. **DATE FORMATTING**: All dates MUST be in format "MMM DD, YYYY" (e.g., "{week_start_str}")

14. Ensure all monetary values are formatted appropriately (e.g., "$1.2B", "$100M")

=== OUTPUT REQUIREMENTS ===
- Return ONLY valid JSON matching the schema above
- Do not include any markdown formatting, code blocks, or explanatory text
- Ensure all numeric fields are numbers (not strings)
- Ensure all string fields are properly escaped
- weekRange MUST be: {{"start": "{week_start_str}", "end": "{week_end_str}"}}
- dailyPerformance MUST include exactly {len(trading_days_info)} entries, one for each trading day listed above
- Include at least 3-5 key highlights (with dates from the week range)
- Include at least 2-3 items in each SWOT category

Generate the JSON now:"""
    
    return prompt


def get_custom_analysis_prompt(symbol: str, timeframe: str = "weekly", focus_areas: List[str] = None) -> str:
    """
    Generate a customizable analysis prompt for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        timeframe: Analysis timeframe ("daily", "weekly", "monthly", "quarterly")
        focus_areas: Optional list of specific areas to focus on:
            - "price_action": Price movements and technical analysis
            - "fundamentals": Financial metrics and earnings
            - "news": News and sentiment
            - "analyst": Analyst opinions and recommendations
            - "comparison": Sector and peer comparisons
    
    Returns:
        Customized prompt string
    """
    timeframe_map = {
        "daily": "today",
        "weekly": "this week",
        "monthly": "this month",
        "quarterly": "this quarter"
    }
    
    time_period = timeframe_map.get(timeframe.lower(), "this week")
    
    base_prompt = f"Provide a {timeframe} market analysis for {symbol} covering {time_period}."
    
    if focus_areas:
        focus_sections = []
        if "price_action" in focus_areas:
            focus_sections.append("price movements, technical indicators, and chart patterns")
        if "fundamentals" in focus_areas:
            focus_sections.append("financial metrics, earnings, and fundamental analysis")
        if "news" in focus_areas:
            focus_sections.append("news events, sentiment, and market developments")
        if "analyst" in focus_areas:
            focus_sections.append("analyst opinions, ratings, and recommendations")
        if "comparison" in focus_areas:
            focus_sections.append("sector comparisons and relative performance")
        
        if focus_sections:
            base_prompt += f"\n\nFocus specifically on: {', '.join(focus_sections)}."
    
    base_prompt += "\n\nUse hybrid search to gather comprehensive information from multiple data sources."
    
    return base_prompt


# Example usage
if __name__ == "__main__":
    # Test with a symbol
    symbol = "AAPL"
    prompt = get_weekly_market_analysis_prompt(symbol)
    print("=" * 70)
    print(f"Weekly Market Analysis Prompt for {symbol}")
    print("=" * 70)
    print(prompt)
    print("\n" + "=" * 70)
    
    # Get all prompts
    print("\nGenerating prompts for all symbols...")
    all_prompts = get_symbol_analysis_prompts()
    print(f"Generated {len(all_prompts)} prompts")
    for sym, prompt_text in all_prompts.items():
        print(f"  - {sym}: {len(prompt_text)} characters")

