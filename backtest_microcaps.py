#!/usr/bin/env python3
"""
DCS v4.6 Backtest on Microcap Stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 20 microcap/small cap stocks
SYMBOLS = [
    'GME', 'AMC', 'BB', 'NOK',  # Meme/small tech
    'SNAP', 'PLTR',  # Growth tech  
    'RIVN', 'LCID', 'SOFI', 'UPST',  # Growth
    'DNA', 'NVAX', 'PCRX', 'APPS',  # Biotech/Tech
    'CTXR', 'ATER', 'CLNE',  # Microcaps
    'BLNK', 'CHPT',  # EV charging
]
TIMEFRAMES = ['1d']  # Daily for stocks
INITIAL_CAPITAL = 10000
FIXED_UNITS = 100  # Share units (stocks, not forex lots)

def calc_indicators(df, p_fast=14, p_mid=21, p_atr=14, p_vol=20):
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    df['emaFast'] = c.ewm(span=p_fast, adjust=False).mean()
    df['emaMid'] = c.ewm(span=p_mid, adjust=False).mean()
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(p_atr).mean()
    df['volAvg'] = v.rolling(p_vol).mean()
    return df

def calc_signals(df, p_swing=20, min_cand=58, min_exec=75):
    c = df['Close'].shift(1)
    o = df['Open'].shift(1)
    h = df['High'].shift(1)
    l = df['Low'].shift(1)
    v = df['Volume'].shift(1)
    emaFast1 = df['emaFast'].shift(1)
    emaMid1 = df['emaMid'].shift(1)
    atr1 = df['atr'].shift(1)
    volAvg1 = df['volAvg'].shift(1)
    rn = h - l
    body = (c - o).abs()
    bodyEff = body / rn.replace(0, np.nan)
    sh = df['High'].shift(2).rolling(p_swing).max()
    sl = df['Low'].shift(2).rolling(p_swing).min()
    vr = v / volAvg1.replace(0, np.nan)
    bull = (c > emaMid1) & (emaFast1 > emaMid1)
    bear = (c < emaMid1) & (emaFast1 < emaMid1)
    momB = (c > o) & (c > emaFast1) & (emaFast1 >= df['emaFast'].shift(2))
    momS = (c < o) & (c < emaFast1) & (emaFast1 <= df['emaFast'].shift(2))
    dispB = (c > o) & (bodyEff >= 0.55) & (rn > atr1 * 0.80)
    dispS = (c < o) & (bodyEff >= 0.55) & (rn > atr1 * 0.80)
    thinP = vr < 0.75
    qualP = vr >= 0.90
    pullB = bull & (l <= emaFast1 + atr1 * 0.35) & (c >= emaFast1) & (c > o)
    pullS = bear & (h >= emaFast1 - atr1 * 0.35) & (c <= emaFast1) & (c < o)
    breakB = (c > sh) & dispB & qualP
    breakS = (c < sl) & dispS & qualP
    candB = pullB | breakB
    candS = pullS | breakS
    bullSc = np.where(bull, 18, np.where(c > emaMid1, 10, 0))
    bearSc = np.where(bear, 18, np.where(c < emaMid1, 10, 0))
    bullSc += np.where(momB, 14, 0) + np.where(dispB, 10, 0) + np.where(bull, 14, 0)
    bearSc += np.where(momS, 14, 0) + np.where(dispS, 10, 0) + np.where(bear, 14, 0)
    bullSc += np.where(qualP & (c > o), 12, np.where(thinP, 0, 6))
    bearSc += np.where(qualP & (c < o), 12, np.where(thinP, 0, 6))
    bullSc += np.where((bull|bear) & bull, 12, 5)
    bearSc += np.where((bull|bear) & bear, 12, 5)
    biasB = bullSc > (bearSc + 8)
    biasS = bearSc > (bullSc + 8)
    entryQB = np.where(pullB, 22, np.where(breakB, 18, 6))
    entryQS = np.where(pullS, 22, np.where(breakS, 18, 6))
    candScB = np.where(bullSc > 0, bullSc + entryQB, 0)
    candScS = np.where(bearSc > 0, bearSc + entryQS, 0)
    riskB = thinP
    riskS = thinP
    execB = np.where(riskB, 0, candScB - 20)
    execS = np.where(riskS, 0, candScS - 20)
    df['appLong'] = candB & biasB & (candScB >= min_cand) & (execB >= min_exec)
    df['appShort'] = candS & biasS & (candScS >= min_cand) & (execS >= min_exec)
    df['atr1'] = atr1
    return df

def backtest_stock(df, target_r=3.0, stop_atr=0.8, cooldown=12):
    """Backtest for stocks"""
    df = df.dropna()
    if len(df) < 50:
        return None
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    cd = 0
    for i in range(50, len(df)):
        if cd > 0:
            cd -= 1
        price = df['Close'].iloc[i]
        atr = df['atr1'].iloc[i]
        if position is None and cd == 0:
            if df['appLong'].iloc[i]:
                entry = price  # stocks don't have spread like forex
                stop = entry - atr * stop_atr
                target = entry + (entry - stop) * target_r
                # Fixed shares
                position = {'type': 'long', 'e': entry, 's': stop, 't': target, 'sz': FIXED_UNITS}
                cd = cooldown
            elif df['appShort'].iloc[i]:
                entry = price
                stop = entry + atr * stop_atr
                target = entry - (stop - entry) * target_r
                position = {'type': 'short', 'e': entry, 's': stop, 't': target, 'sz': FIXED_UNITS}
                cd = cooldown
        elif position:
            exited = False
            pnl = 0
            if position['type'] == 'long':
                if price <= position['s']:
                    pnl = (position['s'] - position['e']) * position['sz']
                    exited = True
                elif price >= position['t']:
                    pnl = (position['t'] - position['e']) * position['sz']
                    exited = True
            else:
                if price >= position['s']:
                    pnl = (position['e'] - position['s']) * position['sz']
                    exited = True
                elif price <= position['t']:
                    pnl = (position['e'] - position['t']) * position['sz']
                    exited = True
            if exited:
                # Commission
                pnl = pnl - (position['sz'] * position['e'] * 0.001)
                capital += pnl
                trades.append(pnl)
                position = None
    if not trades:
        return None
    wins = sum(1 for x in trades if x > 0)
    risk_per_trade = FIXED_UNITS * 0.01 * 20  # rough
    avg_r = np.mean([t / risk_per_trade for t in trades])
    return {
        'trades': len(trades),
        'win_rate': wins/len(trades)*100,
        'avg_r': avg_r,
        'return': (capital-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
    }

print("="*65)
print("DCS v4.6 Backtest - Microcap Stocks (1 Year)")
print("="*65)

results = []
for sym in SYMBOLS:
    try:
        df = yf.download(sym, period='1y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 200:
            print(f"{sym}: No data")
            continue
        df = calc_indicators(df)
        df = calc_signals(df, 20, 60, 75)
        result = backtest_stock(df, 3.0, 0.8, 12)
        if result and result['trades'] > 0:
            results.append(result)
            print(f"{sym}: Trades={result['trades']}, Win%={result['win_rate']:.1f}%, Return={result['return']:.1f}%")
        else:
            print(f"{sym}: No signals")
    except Exception as e:
        print(f"{sym}: Error - {str(e)[:40]}")

print("\n" + "="*65)
print("SUMMARY")
print("="*65)
if results:
    total_trades = sum(r['trades'] for r in results)
    avg_win = np.mean([r['win_rate'] for r in results])
    avg_return = np.mean([r['return'] for r in results])
    avg_r = np.mean([r['avg_r'] for r in results])
    print(f"Stocks tested: {len(results)}")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Win Rate: {avg_win:.1f}%")
    print(f"Avg Return: {avg_return:.1f}%")
    print(f"Avg R: {avg_r:.2f}")