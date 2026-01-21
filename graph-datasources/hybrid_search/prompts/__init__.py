"""
Prompts package for Hybrid Search Agent.

This package contains prompt templates and generators for various analysis tasks.
"""

# Export utility functions
from .utils import (
    load_symbols,
    get_current_trading_week,
    format_date_for_prompt,
    format_day_abbreviation
)

# Export functions from weekly_market_analysis_prompt
from .weekly_market_analysis_prompt import (
    get_weekly_market_analysis_system_prompt,
    get_weekly_market_analysis_user_prompt,
    get_report_type_focus,
    get_weekly_market_analysis_prompt
)

__all__ = [
    # Utility functions
    "load_symbols",
    "get_current_trading_week",
    "format_date_for_prompt",
    "format_day_abbreviation",
    # Weekly market analysis functions
    "get_weekly_market_analysis_system_prompt",
    "get_weekly_market_analysis_user_prompt",
    "get_report_type_focus",
    "get_weekly_market_analysis_prompt",
]

