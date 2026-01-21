"""
Utility functions for prompt generation.

Contains helper functions for date formatting, trading week calculations, etc.
"""

from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import json


def load_symbols() -> List[str]:
    """Load symbols from symbols.json file"""
    symbols_path = Path(__file__).parent.parent / "symbols.json"
    try:
        with open(symbols_path, 'r') as f:
            symbols = json.load(f)
        return symbols if isinstance(symbols, list) else []
    except Exception as e:
        print(f"Warning: Could not load symbols.json: {e}")
        return []


def get_current_trading_week(reference_date: Optional[datetime] = None) -> Tuple[datetime, datetime, List[datetime]]:
    """
    Calculate the trading week (Monday to Friday) for a given date.
    
    Args:
        reference_date: Optional datetime to calculate the trading week for.
                       If None, uses datetime.now()
    
    Returns:
        Tuple of (week_start, week_end, trading_days)
        - week_start: Monday of the week containing reference_date
        - week_end: Friday of that week (or reference_date if before Friday)
        - trading_days: List of all trading days (Mon-Fri) in the week up to reference_date
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Find Monday of the week containing reference_date (weekday: 0=Monday, 6=Sunday)
    days_since_monday = reference_date.weekday()
    week_start = reference_date - timedelta(days=days_since_monday)
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Find Friday of that week (or reference_date if it's before Friday)
    days_until_friday = 4 - days_since_monday  # 4 = Friday (0-indexed)
    if days_until_friday < 0:
        # If it's Saturday/Sunday, use Friday of that week
        week_end = week_start + timedelta(days=4)
    else:
        # Use reference_date if it's Mon-Fri, otherwise use Friday
        week_end = min(reference_date, week_start + timedelta(days=4))
    
    week_end = week_end.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Generate list of trading days (Mon-Fri) up to reference_date
    trading_days = []
    for i in range(5):  # Monday (0) to Friday (4)
        day = week_start + timedelta(days=i)
        if day <= reference_date:  # Only include days up to reference_date
            trading_days.append(day)
    
    return week_start, week_end, trading_days


def format_date_for_prompt(dt: datetime) -> str:
    """Format datetime as 'MMM DD, YYYY' (e.g., 'Oct 14, 2024')"""
    return dt.strftime("%b %d, %Y")


def format_day_abbreviation(dt: datetime) -> str:
    """Format datetime as day abbreviation (e.g., 'Mon', 'Tue')"""
    return dt.strftime("%a")



