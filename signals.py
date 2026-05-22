"""
DCS Engine - Signal Generation
Generates trading signals based on 1st touch of broken S/R levels
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional

def detect_touch(price: float, level: float, high: float, low: float, 
                 touch_distance: float = 0.0015) -> bool:
    """Check if price touches a level"""
    # Check if level is within price range
    if low <= level <= high:
        return True
    # Check if price is within distance of level
    if abs(price - level) / price < touch_distance:
        return True
    return False

def generate_signals(df: pd.DataFrame, breaks_up: List[Dict], breaks_down: List[Dict],
                    touch_distance: float = 0.0015) -> pd.DataFrame:
    """
    Generate signals on 1st touch of broken levels
    """
    signals = pd.DataFrame(index=df.index)
    signals['long'] = False
    signals['short'] = False
    signals['level'] = None
    
    # Track touch counts
    for up in breaks_up:
        up['touches'] = 0
    for down in breaks_down:
        down['touches'] = 0
    
    for i in range(50, len(df)):
        price = df['Close'].iloc[i]
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        
        # Check long signals (touch broken resistance = support)
        for up in breaks_up:
            if up['touches'] == 0 and detect_touch(price, up['level'], high, low, touch_distance):
                signals.iloc[i, signals.columns.get_loc('long')] = True
                signals.iloc[i, signals.columns.get_loc('level')] = up['level']
                up['touches'] += 1
                break
        
        # Check short signals (touch broken support = resistance)
        for down in breaks_down:
            if down['touches'] == 0 and detect_touch(price, down['level'], high, low, touch_distance):
                signals.iloc[i, signals.columns.get_loc('short')] = True
                signals.iloc[i, signals.columns.get_loc('level')] = down['level']
                down['touches'] += 1
                break
    
    return signals

def get_signal_stats(signals: pd.DataFrame) -> Dict:
    """Get signal statistics"""
    long_signals = signals['long'].sum()
    short_signals = signals['short'].sum()
    
    return {
        'total_long': int(long_signals),
        'total_short': int(short_signals),
        'total_signals': int(long_signals + short_signals)
    }
