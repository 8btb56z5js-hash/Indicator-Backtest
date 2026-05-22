#!/usr/bin/env python3
"""
Simple backtest for DCS-style indicator
Fixed to actually generate trades
"""

import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np

SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
TIMEFRAMES = ['1h', '4h', '1d']
INITIAL_CAPITAL = 10000
RISK_PCT = 0.02

def calculate_indicators(df):
    """Calculate DCS-style indicators"""
    # EMAs
    df['emaFast'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['emaMid'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['emaSlow'] = df['Close'].ewm(span=55, adjust=False).mean()
    
    # ATR
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # ATR bands
    df['atrPct'] = df['atr'] / df['Close']
    df['bandMult'] = np.select(
        [df['atrPct'] > 0.08, df['atrPct'] > 0.04, df['atrPct'] > 0.02],
        [2.8, 2.4, 2.1], default=1.8
    )
    df['upperBand'] = df['emaMid'] + df['atr'] * df['bandMult']
    df['lowerBand'] = df['emaMid'] - df['atr'] * df['bandMult']
    
    # Volume
    df['volAvg'] = df['Volume'].rolling(20).mean()
    
    return df

def generate_signals(df):
    """Generate simplified DCS signals"""
    # Previous bar data
    df['c1'] = df['Close'].shift(1)
    df['o1'] = df['Open'].shift(1)
    df['h1'] = df['High'].shift(1)
    df['l1'] = df['Low'].shift(1)
    
    # Trend detection
    df['bullTrend'] = (df['c1'] > df['emaMid']) & (df['emaFast'] > df['emaMid'])
    df['bearTrend'] = (df['c1'] < df['emaMid']) & (df['emaFast'] < df['emaMid'])
    
    # Setup patterns (simplified)
    df['pullbackLong'] = df['bullTrend'] & (df['l1'] <= df['emaFast'] + df['atr'] * 0.35) & (df['c1'] >= df['emaFast']) & (df['c1'] > df['o1'])
    df['pullbackShort'] = df['bearTrend'] & (df['h1'] >= df['emaFast'] - df['atr'] * 0.35) & (df['c1'] <= df['emaFast']) & (df['c1'] < df['o1'])
    
    # Breakout
    df['swingHigh'] = df['High'].shift(1).rolling(20).max()
    df['swingLow'] = df['Low'].shift(1).rolling(20).min()
    df['breakoutLong'] = (df['c1'] > df['swingHigh']) & df['bullTrend']
    df['breakoutShort'] = (df['c1'] < df['swingLow']) & df['bearTrend']
    
    # Combined signals
    df['longSignal'] = df['pullbackLong'] | df['breakoutLong']
    df['shortSignal'] = df['pullbackShort'] | df['breakoutShort']
    
    return df

def run_backtest(df, symbol, tf):
    """Run backtest with proper position management"""
    df = df.dropna()
    if len(df) < 50:
        return None
    
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    equity = [capital]
    
    for i in range(50, len(df)):
        price = df['Close'].iloc[i]
        
        if position is None:
            # Entry signals
            if df['longSignal'].iloc[i]:
                stop = price - df['atr'].iloc[i] * 0.45
                target = price + (price - stop) * 1.8
                position = {
                    'type': 'long',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'size': (capital * RISK_PCT) / (price - stop)
                }
            elif df['shortSignal'].iloc[i]:
                stop = price + df['atr'].iloc[i] * 0.45
                target = price - (stop - price) * 1.8
                position = {
                    'type': 'short',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'size': (capital * RISK_PCT) / (stop - price)
                }
        else:
            # Exit logic
            exited = False
            pnl = 0
            
            if position['type'] == 'long':
                if price <= position['stop']:
                    pnl = (position['stop'] - position['entry']) * position['size']
                    exited = True
                elif price >= position['target']:
                    pnl = (position['target'] - position['entry']) * position['size']
                    exited = True
            else:
                if price >= position['stop']:
                    pnl = (position['entry'] - position['stop']) * position['size']
                    exited = True
                elif price <= position['target']:
                    pnl = (position['entry'] - position['target']) * position['size']
                    exited = True
            
            if exited:
                capital += pnl
                trades.append({'pnl': pnl, 'type': position['type']})
                equity.append(capital)
                position = None
    
    if len(trades) == 0:
        return None
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    max_dd = 0
    peak = equity[0]
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return {
        'symbol': symbol,
        'tf': tf,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_R': np.mean([t['pnl'] / (INITIAL_CAPITAL * RISK_PCT) for t in trades]) if trades else 0,
        'max_dd': max_dd,
        'final_capital': capital,
        'return_pct': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    }

print("="*60)
print("DCS v4.6 MODIFIED - Backtest Results")
print("="*60)

results = []
for tf in TIMEFRAMES:
    print(f"\n=== Timeframe: {tf.upper()} ===")
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, start=datetime(2025,5,22), end=datetime(2026,5,22), interval=tf, progress=False)
            if df.empty or len(df) < 100:
                print(f"{sym}: No data")
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = calculate_indicators(df)
            df = generate_signals(df)
            
            result = run_backtest(df, sym, tf)
            if result:
                results.append(result)
                print(f"{sym}: Trades={result['trades']}, Win%={result['win_rate']:.1f}%, AvgR={result['avg_R']:.2f}, MaxDD={result['max_dd']:.1f}%, Return={result['return_pct']:.1f}%")
            else:
                print(f"{sym}: No signals")
        except Exception as e:
            print(f"{sym}: Error - {str(e)[:50]}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if results:
    total_trades = sum(r['trades'] for r in results)
    avg_win = np.mean([r['win_rate'] for r in results])
    avg_return = np.mean([r['return_pct'] for r in results])
    avg_dd = np.mean([r['max_dd'] for r in results])
    print(f"Total Trades: {total_trades}")
    print(f"Avg Win Rate: {avg_win:.1f}%")
    print(f"Avg Return: {avg_return:.1f}%")
    print(f"Avg Max DD: {avg_dd:.1f}%")
    print(f"Instruments tested: {len(results)}/{len(SYMBOLS) * len(TIMEFRAMES)}")