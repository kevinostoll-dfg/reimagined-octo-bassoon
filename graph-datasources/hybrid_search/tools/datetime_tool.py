"""
Datetime Tool for LlamaIndex Agent.

This tool provides temporal awareness to the agent, allowing it to understand
the current date, trading week, and time context for intelligent query filtering.
"""

import os
import sys
import json
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from prompts import get_current_trading_week, format_date_for_prompt, format_day_abbreviation

logger = logging.getLogger(__name__)


def get_current_datetime_info() -> str:
    """
    Get current date and time information for temporal awareness.
    
    Returns:
        Formatted string with current date, trading week, and trading days
    """
    try:
        now = datetime.now()
        week_start, week_end, trading_days = get_current_trading_week()
        
        # Format dates
        current_date_str = format_date_for_prompt(now)
        current_time_str = now.strftime("%H:%M:%S")
        week_start_str = format_date_for_prompt(week_start)
        week_end_str = format_date_for_prompt(week_end)
        
        # Format trading days
        trading_days_info = []
        for day in trading_days:
            trading_days_info.append({
                "day": format_day_abbreviation(day),
                "date": format_date_for_prompt(day),
                "iso_date": day.strftime("%Y-%m-%d")
            })
        
        # Build response
        info = f"""CURRENT DATE AND TIME INFORMATION:

Current Date: {current_date_str}
Current Time: {current_time_str}
Current ISO Date: {now.strftime("%Y-%m-%d")}
Current ISO DateTime: {now.strftime("%Y-%m-%d %H:%M:%S")}

CURRENT TRADING WEEK:
Week Start: {week_start_str} ({week_start.strftime("%Y-%m-%d")})
Week End: {week_end_str} ({week_end.strftime("%Y-%m-%d")})
Week Range: {week_start_str} to {week_end_str}

TRADING DAYS THIS WEEK ({len(trading_days_info)} days):
"""
        for day_info in trading_days_info:
            info += f"  - {day_info['day']}, {day_info['date']} ({day_info['iso_date']})\n"
        
        info += f"""
TEMPORAL CONTEXT FOR QUERIES:
- When querying for "this week" or "weekly" data, use dates from {week_start_str} to {week_end_str}
- ISO date format (YYYY-MM-DD) should be used for date filtering in metadata queries
- Current week includes {len(trading_days_info)} trading days
- Today is {now.strftime("%A, %B %d, %Y")}
"""
        
        return info
        
    except Exception as e:
        import traceback
        logger.error(f"Error getting datetime info: {e}")
        logger.error(traceback.format_exc())
        return f"Error getting datetime information: {repr(e)}"


def get_trading_week_range() -> str:
    """
    Get the current trading week date range in multiple formats.
    
    Returns:
        Formatted string with week range in various date formats
    """
    try:
        week_start, week_end, trading_days = get_current_trading_week()
        
        week_start_str = format_date_for_prompt(week_start)
        week_end_str = format_date_for_prompt(week_end)
        week_start_iso = week_start.strftime("%Y-%m-%d")
        week_end_iso = week_end.strftime("%Y-%m-%d")
        
        return f"""CURRENT TRADING WEEK RANGE:

Formatted Dates: {week_start_str} to {week_end_str}
ISO Dates: {week_start_iso} to {week_end_iso}
Date Range for Queries: {week_start_iso} to {week_end_iso}

Use these dates when filtering queries for "this week" or "weekly" data.
For Milvus metadata filters, use ISO format: metadata["date"] >= "{week_start_iso}" and metadata["date"] <= "{week_end_iso}"
"""
        
    except Exception as e:
        logger.error(f"Error getting trading week range: {e}")
        return f"Error getting trading week range: {repr(e)}"


# Create the Tool objects
from llama_index.core.tools import FunctionTool

datetime_info_tool = FunctionTool.from_defaults(
    fn=get_current_datetime_info,
    name="get_current_datetime_info",
    description="""Get current date, time, and trading week information for temporal awareness.
    
    This tool provides the agent with current temporal context, including:
    - Current date and time (multiple formats)
    - Current trading week range (Monday to Friday)
    - List of trading days in the current week
    - Date formats for use in queries
    
    WHEN TO USE THIS TOOL:
    USE when query asks for:
       - "this week", "current week", "this month", "today", "now", "latest", "recent"
       - "weekly report", "current market", "latest news", "recent earnings"
       - Time-sensitive information requiring current date context
       - Reports or analysis that need to reference "current" or "latest" dates
    
    DO NOT use when query asks for:
       - "historical", "past", "previous", "all time", "all data"
       - Specific dates/years: "2025", "Q1 2025", "January 2025", "last year"
       - "all earnings", "all transcripts", "all statements"
       - Queries explicitly asking for historical or all-time data without date restrictions
    
    The tool automatically:
    - Calculates the current trading week (Monday to Friday)
    - Provides dates in multiple formats (formatted and ISO)
    - Lists all trading days in the current week
    - Provides guidance on using dates in queries
    
    Returns:
        Comprehensive date/time information including:
        - Current date and time
        - Trading week range
        - Trading days list
        - Date format guidance for queries
    
    Example usage:
    - Call this tool when generating weekly reports or time-sensitive queries
    - Use the returned dates to filter Milvus queries by date range
    - Use the trading week range for Graph queries about "this week"
    - Reference current date when generating time-sensitive analysis
    
    Note: For historical queries, skip this tool and query without date restrictions.
    """
)

trading_week_range_tool = FunctionTool.from_defaults(
    fn=get_trading_week_range,
    name="get_trading_week_range",
    description="""Get the current trading week date range for filtering queries.
    
    WHEN TO USE THIS TOOL:
    USE when you need date ranges for filtering "this week" or "current week" queries
    USE when generating weekly reports that need current week date boundaries
    USE when Milvus/Graph queries need ISO date format filters for current week
    
    DO NOT use for historical queries - query without date restrictions instead
    
    Returns ISO date range (YYYY-MM-DD format) suitable for metadata filters.
    
    This tool provides the trading week date range in multiple formats,
    specifically designed for use in date-filtered queries.
    
    Use this tool when:
    - You need date ranges for Milvus metadata filters
    - You need to filter Graph queries by "this week"
    - You're building date-filtered queries
    - You need ISO date format for database queries
    
    Returns:
        Trading week range in formatted and ISO date formats,
        with guidance on using dates in queries.
    """
)

if __name__ == "__main__":
    # Test the tools
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Datetime Tools ===\n")
    print("1. Current Datetime Info:")
    print(get_current_datetime_info())
    print("\n2. Trading Week Range:")
    print(get_trading_week_range())
    print("\n=== Test Complete ===")

