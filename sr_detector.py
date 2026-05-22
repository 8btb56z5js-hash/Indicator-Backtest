"""
DCS Engine - S/R Level Detection
Detects broken support/resistance levels from HTF data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def get_htf_data(symbol: str, timeframe: str = "4h", period: str = "2y") -> pd.DataFrame:
    """Fetch higher timeframe data"""
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=timeframe)

def detect_swings(df: pd.DataFrame, lookback: int = 2) -> Tuple[list, list]:
    """Detect swing highs and lows"""
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        
        # Check swing high
        if all(df['High'].iloc[i-lookback:i] < high) and all(df['High'].iloc[i+1:i+lookback+1] <= high):
            highs.append({'price': high, 'index': i, 'time': df.index[i]})
        
        # Check swing low
        if all(df['Low'].iloc[i-lookback:i] > low) and all(df['Low'].iloc[i+1:i+lookback+1] >= low):
            lows.append({'price': low, 'index': i, 'time': df.index[i]})
    
    return highs, lows

def detect_breaks(df_htf: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect broken S/R levels:
    - Resistance broken UP -> becomes support
    - Support broken DOWN -> becomes resistance
    """
    breaks_up = []   # Broken resistance (now support)
    breaks_down = []  # Broken support (now resistance)
    
    htf_close = df_htf['Close'].values
    htf_high = df_htf['High'].values
    htf_low = df_htf['Low'].values
    
    for i in range(3, len(df_htf)):
        prev_high = htf_high[i-2]
        prev_low = htf_low[i-2]
        curr_close = htf_close[i]
        prev_close = htf_close[i-1]
        
        # Resistance broken UP -> new support
        if curr_close > prev_high and prev_close < prev_high:
            breaks_up.append({'level': prev_high, 'time': df_htf.index[i], 'touches': 0})
        
        # Support broken DOWN -> new resistance  
        if curr_close < prev_low and prev_close > prev_low:
            breaks_down.append({'level': prev_low, 'time': df_htf.index[i], 'touches': 0})
    
    return breaks_up, breaks_down

def get_current_levels(breaks_up: List[Dict], breaks_down: List[Dict], 
                       current_price: float, touch_distance: float = 0.0015) -> Dict:
    """Get active S/R levels near current price"""
    active_support = [b for b in breaks_up if abs(current_price - b['level']) / current_price < touch_distance]
    active_resistance = [b for b in breaks_down if abs(current_price - b['level']) / current_price < touch_distance]
    
    return {
        'support_levels': active_support,
        'resistance_levels': active_resistance
    }
