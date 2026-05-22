#!/usr/bin/env python3
"""
HTF S/R + 1st Touch Momentum Backtest v5
- 2:1 Risk/Reward (8 pip TP, 4 pip stop)
- Testing different pip configurations
"""

import yfinance as yf
import pandas as pd
import numpy as np

SYMBOL = "EURUSD=X"
INITIAL_CAPITAL = 10000
POSITION_SIZE = 0.02
RSI_PERIOD = 14
RSI_OS = 40
RSI_OB = 60
TOUCH_DISTANCE = 0.0015

# Pip configs to test
PIP_CONFIGS = [
    (4, 8, "4:8 (2:1)"),
    (3, 9, "3:9 (3:1)"),
    (5, 10, "5:10 (2:1)"),
    (3, 6, "3:6 (2:1)"),
]

def get_data(interval="1d"):
    return yf.Ticker(SYMBOL).history(period="2y", interval=interval)

def detect_breaks(df_htf, label=""):
    breaks_up, breaks_down = [], []
    for i in range(3, len(df_htf)):
        prev_high, prev_low = df_htf['High'].iloc[i-2], df_htf['Low'].iloc[i-2]
        curr_close, prev_close = df_htf['Close'].iloc[i], df_htf['Close'].iloc[i-1]
        if curr_close > prev_high and prev_close < prev_high:
            breaks_up.append({'level': prev_high, 'touches': 0, 'tf': label})
        if curr_close < prev_low and prev_close > prev_low:
            breaks_down.append({'level': prev_low, 'touches': 0, 'tf': label})
    return breaks_up, breaks_down

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def run_backtest(breaks_up, breaks_down, stop_pips, tp_pips, label):
    df_daily = get_data("1d")
    df_daily['RSI'] = calculate_rsi(df_daily['Close'], RSI_PERIOD)
    
    STOP_PIPS = stop_pips * 0.0001
    TP_PIPS = tp_pips * 0.0001
    EXIT_BARS = 15
    
    capital, trades, position = INITIAL_CAPITAL, [], None
    
    for i in range(50, len(df_daily)):
        bar, price, rsi = df_daily.iloc[i], df_daily['Close'].iloc[i], df_daily['RSI'].iloc[i]
        if pd.isna(rsi): continue
        
        # Entry logic
        if not position:
            for lvl in breaks_up:
                if (abs(price - lvl['level']) < TOUCH_DISTANCE or bar['Low'] <= lvl['level'] <= bar['High']) and lvl['touches'] == 0 and rsi <= RSI_OS:
                    position = {'type': 'long', 'entry': price, 'stop': price - STOP_PIPS, 'target': price + TP_PIPS, 'entry_bar': i}
                    lvl['touches'] = 1
                    break
            if not position:
                for lvl in breaks_down:
                    if (abs(price - lvl['level']) < TOUCH_DISTANCE or bar['Low'] <= lvl['level'] <= bar['High']) and lvl['touches'] == 0 and rsi >= RSI_OB:
                        position = {'type': 'short', 'entry': price, 'stop': price + STOP_PIPS, 'target': price - TP_PIPS, 'entry_bar': i}
                        lvl['touches'] = 1
                        break
        
        # Exit logic
        if position:
            bars = i - position['entry_bar']
            exit_price, reason = price, ""
            if (position['type'] == 'long' and price <= position['stop']) or (position['type'] == 'short' and price >= position['stop']):
                exit_price, reason = position['stop'], "stop"
            elif (position['type'] == 'long' and price >= position['target']) or (position['type'] == 'short' and price <= position['target']):
                exit_price, reason = position['target'], "tp"
            elif bars >= EXIT_BARS:
                exit_price, reason = price, "time"
            
            if reason:
                pnl = (exit_price - position['entry']) / position['entry'] * capital if position['type'] == 'long' else (position['entry'] - exit_price) / position['entry'] * capital
                capital += pnl
                trades.append({'type': position['type'], 'entry': position['entry'], 'exit': exit_price, 'pnl': pnl, 'reason': reason})
                position = None
    
    if position:
        final = df_daily['Close'].iloc[-1]
        pnl = (final - position['entry']) / position['entry'] * capital if position['type'] == 'long' else (position['entry'] - final) / position['entry'] * capital
        capital += pnl
    
    return capital, trades

# Get 4H data
df_4h = get_data("4h")
breaks_up, breaks_down = detect_breaks(df_4h, "4H")
print(f"4H levels: {len(breaks_up)} up, {len(breaks_down)} down\n")

print("="*60)
print("BACKTEST v5 - 2:1 Risk/Reward on 4H")
print("="*60)

for stop_pips, tp_pips, name in PIP_CONFIGS:
    # Reset touch counts
    for b in breaks_up + breaks_down: b['touches'] = 0
    
    capital, trades = run_backtest(breaks_up[:], breaks_down[:], stop_pips, tp_pips, name)
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    stops = len([t for t in trades if t['reason'] == 'stop'])
    tps = len([t for t in trades if t['reason'] == 'tp'])
    
    ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    winrate = len(wins)/len(trades)*100 if trades else 0
    
    print(f"{name}: Trades={len(trades)} | Return={ret:.2f}% | Win={winrate:.1f}% | Stops={stops} | TPs={tps}")