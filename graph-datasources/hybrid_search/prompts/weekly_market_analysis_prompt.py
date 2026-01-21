"""
Weekly Market Analysis Prompt for LlamaIndex Agent.

This module provides the system prompt for generating comprehensive weekly US equity market reports.
The prompt guides the agent to use its tools (milvus_search, graph_search, tavily_search, FMP tools)
to gather data and generate professional-grade weekly market analysis reports.

Based on GitHub Issue #2781.
"""

from typing import Optional, Literal
from datetime import datetime

# Report types
ReportType = Literal[
    "total_us_market",
    "large_cap", 
    "mid_cap",
    "small_cap",
    "tech_mag7",
    "ai_sector",
    "by_sector"
]


def get_weekly_market_analysis_system_prompt() -> str:
    """
    Returns the minimal system prompt for weekly market analysis.
    Contains only tool descriptions and basic ReAct format - kept minimal.
    """
    return """You are a senior US equity strategist with access to financial data tools.

AVAILABLE TOOLS:
{tool_desc}

TOOL NAMES: {tool_names}

REACT FORMAT (MANDATORY):
Thought: [Reasoning]
Action: <tool_name>
Action Input: <JSON>
Observation: [result]
[Repeat until you have all data]
Final Answer: [Write report ONLY after gathering data]

KEY PRINCIPLES:
- ALWAYS use Thought → Action → Observation format
- NEVER skip tool calls - gather data before writing
- Only write Final Answer after gathering all data"""


def get_report_type_focus(report_type: ReportType) -> str:
    """Get focus instructions based on report type."""
    focus_map = {
        "total_us_market": """
Focus on:
- All major US equity indices (S&P 500, Nasdaq 100, Dow, Russell 2000)
- Complete market-cap spectrum (mega/large, mid, small cap)
- All 11 GICS sectors with equal weight
- Broad market themes and macro drivers
""",
        "large_cap": """
Focus on:
- S&P 500 and large-cap indices
- Mega-cap and large-cap stocks (typically >$10B market cap)
- Large-cap sector performance and rotation
- Institutional flows and positioning
- Blue-chip earnings and guidance
""",
        "mid_cap": """
Focus on:
- Mid-cap indices (S&P MidCap 400, etc.)
- Mid-cap stocks (typically $2B-$10B market cap)
- Mid-cap sector performance
- Growth opportunities and M&A activity
- Mid-cap earnings trends
""",
        "small_cap": """
Focus on:
- Small-cap indices (Russell 2000, etc.)
- Small-cap stocks (typically <$2B market cap)
- Small-cap sector performance
- Risk appetite and liquidity conditions
- Small-cap earnings and guidance
- Small-cap vs large-cap relative performance
""",
        "tech_mag7": """
Focus on:
- Magnificent 7 stocks: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- Technology sector performance
- Nasdaq 100 and tech-heavy indices
- AI and innovation themes
- Tech earnings and guidance
- Tech sector valuation and multiples
- Concentration risk in mega-cap tech
""",
        "ai_sector": """
Focus on:
- AI-related stocks and themes
- Semiconductors (NVDA, AMD, etc.)
- Cloud and infrastructure (MSFT, AMZN, GOOGL)
- AI applications and software
- AI capex and investment trends
- Regulatory and policy impacts on AI
- AI sector valuation and momentum
""",
        "by_sector": """
Focus on:
- Individual sector deep-dive (sector specified in query)
- Sector-specific fundamentals
- Sector earnings and guidance
- Sector regulatory and policy environment
- Sector SWOT analysis
- Sector relative performance vs market
"""
    }
    return focus_map.get(report_type, focus_map["total_us_market"])


def get_weekly_market_analysis_user_prompt(
    report_type: ReportType = "total_us_market",
    sector: Optional[str] = None,
    as_of_date: Optional[str] = None,
    focus_themes: Optional[str] = None,
    specific_tickers: Optional[str] = None
) -> str:
    """
    Generate an enriched user prompt for weekly market analysis with parameters.
    
    This function returns the detailed user prompt with all weekly market analysis instructions,
    workflow, and report structure. This should be used as the user query/instruction.
    
    Args:
        report_type: Type of report to generate (default: "total_us_market")
        sector: Sector name for "by_sector" report type (e.g., "Technology", "Financials")
        as_of_date: Analysis date in format "YYYY-MM-DD" (defaults to today)
        focus_themes: Optional themes to emphasize (e.g., "AI earnings upside", "soft landing")
        specific_tickers: Optional comma-separated tickers to emphasize (e.g., "AAPL,MSFT,TSLA")
    
    Returns:
        Enriched user prompt string with all weekly market analysis instructions
    """
    from .utils import get_current_trading_week, format_date_for_prompt
    
    # Calculate date range
    if as_of_date:
        try:
            analysis_date = datetime.strptime(as_of_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD format.")
    else:
        analysis_date = datetime.now()
    
    # Calculate trading week based on the analysis_date (not today)
    week_start, week_end, trading_days = get_current_trading_week(reference_date=analysis_date)
    week_start_str = format_date_for_prompt(week_start)
    week_end_str = format_date_for_prompt(week_end)
    as_of_date_str = format_date_for_prompt(analysis_date)
    
    # ISO dates for filters
    week_start_iso = week_start.strftime("%Y-%m-%d")
    week_end_iso = week_end.strftime("%Y-%m-%d")
    as_of_date_iso = analysis_date.strftime("%Y-%m-%d")
    
    # Validate report type
    valid_types = ["total_us_market", "large_cap", "mid_cap", "small_cap", "tech_mag7", "ai_sector", "by_sector"]
    if report_type not in valid_types:
        raise ValueError(f"Invalid report_type. Must be one of: {', '.join(valid_types)}")
    
    # Validate sector for by_sector type
    if report_type == "by_sector" and not sector:
        raise ValueError("'sector' parameter is required when report_type is 'by_sector'")
    
    # Build enriched user prompt
    user_prompt = f"""TASK: Write a weekly US equity market report (4,000–5,000 words) covering {week_start_str} to {week_end_str}.

CRITICAL: Use ALL tools from ALL three categories before writing. DO NOT generate or make up data.

MANDATORY TOOLS (use all - order flexible):
- FMP: get_company_profile, get_market_data, get_financial_statements, get_financial_ratios, get_analyst_estimates
- Search: graph_search, milvus_search
- News: tavily_search, get_stock_news

WORKFLOW:
1. Gather financial data (FMP tools)
2. Gather context (graph_search, milvus_search)
3. Gather news (tavily_search, get_stock_news)
4. Analyze & synthesize
5. Write report

REPORT STRUCTURE:
1. EXECUTIVE SUMMARY (5–8 paragraphs)
2. MACRO & POLICY BACKDROP
3. US EQUITY MARKET OVERVIEW (S&P 500, Nasdaq, Dow, Russell 2000)
4. SECTOR ANALYSIS WITH SWOT (top 3–5 sectors)
5. EARNINGS & STOCK HIGHLIGHTS (3–7 case studies)
6. RISK ANALYSIS
7. OUTLOOK & POSITIONING (1–4 weeks)
8. APPENDIX

STYLE: Professional, analytical. Cite sources: (Source: [type], YYYY-MM-DD).

DATE FILTERS:
- Milvus: metadata["date"] >= "{week_start_iso}" and metadata["date"] <= "{week_end_iso}"
- Graph: {week_start_iso} to {week_end_iso}
- Ticker: metadata["ticker"] == "SYMBOL" and metadata["date"] >= "{week_start_iso}"

Report Type: {report_type.replace('_', ' ').title()}
{get_report_type_focus(report_type).strip()}"""
    
    if sector:
        user_prompt += f"\nSector: {sector}"
    if focus_themes:
        user_prompt += f"\nThemes: {focus_themes}"
    if specific_tickers:
        user_prompt += f"\nFocus tickers: {specific_tickers}"
    
    user_prompt += "\n\nGenerate report after using ALL tools from ALL categories."
    
    return user_prompt


def get_weekly_market_analysis_prompt(
    report_type: ReportType = "total_us_market",
    sector: Optional[str] = None,
    as_of_date: Optional[str] = None,
    focus_themes: Optional[str] = None,
    specific_tickers: Optional[str] = None
) -> tuple[str, str]:
    """
    Generate both system and user prompts for weekly market analysis.
    
    Returns a tuple of (system_prompt, user_prompt) where:
    - system_prompt: Minimal system prompt with tool descriptions and ReAct format
    - user_prompt: Enriched user prompt with all weekly market analysis instructions
    
    Args:
        report_type: Type of report to generate (default: "total_us_market")
        sector: Sector name for "by_sector" report type
        as_of_date: Analysis date in format "YYYY-MM-DD" (defaults to today)
        focus_themes: Optional themes to emphasize
        specific_tickers: Optional comma-separated tickers to emphasize
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_weekly_market_analysis_system_prompt()
    user_prompt = get_weekly_market_analysis_user_prompt(
        report_type=report_type,
        sector=sector,
        as_of_date=as_of_date,
        focus_themes=focus_themes,
        specific_tickers=specific_tickers
    )
    return (system_prompt, user_prompt)

