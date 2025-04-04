"""
Utility functions for the AI Task Manager.
"""

import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def validate_date(date_str: Optional[str]) -> Optional[datetime.datetime]:
    """
    Validates and converts a date string to a datetime object.
    Accepts formats: YYYY-MM-DD or MM/DD/YYYY
    
    Args:
        date_str: Date string to validate
        
    Returns:
        datetime object if valid, None otherwise
    """
    if not date_str:
        return None
    
    formats = ["%Y-%m-%d", "%m/%d/%Y"]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Invalid date format: {date_str}. Use YYYY-MM-DD or MM/DD/YYYY.")
    return None

def format_date(date_obj: Optional[datetime.datetime]) -> str:
    """
    Format a datetime object into a user-friendly string.
    
    Args:
        date_obj: datetime object to format
        
    Returns:
        Formatted date string
    """
    if not date_obj:
        return "No date"
    
    today = datetime.datetime.now().date()
    date = date_obj.date()
    
    # Calculate difference in days
    delta = date - today
    days_diff = delta.days
    
    # Format based on how far in the future/past
    if days_diff == 0:
        return "Today"
    elif days_diff == 1:
        return "Tomorrow"
    elif days_diff == -1:
        return "Yesterday"
    elif -7 < days_diff < 0:
        return f"{abs(days_diff)} days ago"
    elif 0 < days_diff < 7:
        return f"In {days_diff} days"
    else:
        return date_obj.strftime("%Y-%m-%d")

def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """
    Return singular or plural form of a word based on count.
    
    Args:
        count: Count to check
        singular: Singular form of the word
        plural: Plural form (if None, adds 's' to singular)
        
    Returns:
        Appropriate form of the word
    """
    if not plural:
        plural = singular + 's'
    
    return singular if count == 1 else plural

def get_urgency_color(due_date: Optional[datetime.datetime]) -> str:
    """
    Returns a color code based on urgency of a due date.
    For use in terminal color formatting.
    
    Args:
        due_date: Due date to check
        
    Returns:
        ANSI color code string
    """
    if not due_date:
        return "\033[0m"  # Default color
    
    today = datetime.datetime.now().date()
    date = due_date.date()
    
    days_diff = (date - today).days
    
    if days_diff < 0:
        return "\033[91m"  # Red (overdue)
    elif days_diff == 0:
        return "\033[93m"  # Yellow (due today)
    elif days_diff <= 2:
        return "\033[95m"  # Magenta (due soon)
    else:
        return "\033[0m"  # Default color

def strip_ansi_codes(text: str) -> str:
    """
    Remove ANSI color codes from a string.
    Useful for saving colored text to a file.
    
    Args:
        text: Text with ANSI codes
        
    Returns:
        Text without ANSI codes
    """
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)
