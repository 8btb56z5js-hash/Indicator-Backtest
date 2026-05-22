"""
DCS Engine - Risk Management
Position sizing, stops, and risk management
"""

import pandas as pd
from typing import Dict, Optional

def calculate_position_size(account_size: float, risk_pct: float, 
                           entry: float, stop: float) -> float:
    """Calculate position size based on risk"""
    risk_amount = account_size * risk_pct / 100
    stop_distance = abs(entry - stop) / entry
    position = risk_amount / stop_distance
    return position

def calculate_stop(entry: float, direction: str, stop_pips: int = 4) -> float:
    """Calculate stop loss in price terms"""
    pip_value = 0.0001 * entry
    if direction == 'long':
        return entry - (stop_pips * pip_value)
    else:
        return entry + (stop_pips * pip_value)

def calculate_tp(entry: float, direction: str, tp_pips: int = 8) -> float:
    """Calculate take profit in price terms"""
    pip_value = 0.0001 * entry
    if direction == 'long':
        return entry + (tp_pips * pip_value)
    else:
        return entry - (tp_pips * pip_value)

def apply_risk_management(trades: list, max_daily_loss: float = 2.0) -> list:
    """Filter trades based on risk limits"""
    daily_pnl = 0
    filtered = []
    
    for trade in trades:
        if abs(daily_pnl) >= max_daily_loss:
            break
        filtered.append(trade)
        daily_pnl += trade.get('pnl', 0)
    
    return filtered

def calculate_rr_ratio(entry: float, stop: float, target: float, direction: str) -> float:
    """Calculate risk/reward ratio"""
    if direction == 'long':
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    
    return reward / risk if risk > 0 else 0
