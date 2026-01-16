#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone
from functools import wraps

def strip_timezone(dt):
    """Convert datetime to naive UTC datetime"""
    if dt is None:
        return dt
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # Convert to UTC then remove tzinfo
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def ensure_naive_datetimes(func):
    """
    Decorator to ensure all datetime objects passed to a function are timezone-naive.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Process args
        new_args = []
        for arg in args:
            if isinstance(arg, datetime):
                arg = strip_timezone(arg)
            new_args.append(arg)
        
        # Process kwargs
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                value = strip_timezone(value)
            new_kwargs[key] = value
        
        # Call the original function with sanitized datetimes
        return func(self, *new_args, **new_kwargs)
    return wrapper

def safe_datetime_diff(dt1, dt2):
    """Safely calculate time difference between two datetime objects in seconds"""
    dt1 = strip_timezone(dt1)
    dt2 = strip_timezone(dt2)
    return (dt1 - dt2).total_seconds()