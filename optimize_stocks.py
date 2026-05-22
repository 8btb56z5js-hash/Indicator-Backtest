#!/usr/bin/env python3
"""
Parameter Optimization for Microcap Stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = ['GME', 'AMC', 'BB', 'NOK', 'SNAP', 'PLTR', 'RIVN', 'LCID', 'SOFI', 'UPST', 
           'DNA', 'NVAX', 'PCRX', 'APPS', 'CTXR', 'ATER', 'CLNE', 'BLNK', 'CHPT']
INITIAL_CAPITAL = 10000
FIXED_UNITS = 100

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
                entry = price
                stop = entry - atr * stop_atr
                target = entry + (entry - stop) * target_r
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
                pnl = pnl - (position['sz'] * position['e'] * 0.001)
                capital += pnl
                trades.append(pnl)
                position = None
    if not trades:
        return None
    wins = sum(1 for x in trades if x > 0)
    return {
        'trades': len(trades),
        'win_rate': wins/len(trades)*100,
        'return': (capital-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
    }

# Load all stock data
print("Loading stock data...")
data = {}
for sym in SYMBOLS:
    try:
        df = yf.download(sym, period='1y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 200:
            data[sym] = df
    except:
        pass
print(f"Loaded {len(data)} stocks\n")

# Grid search
results = []
configs = [
    # (min_exec, min_cand, target_r, stop_atr, cooldown)
    (65, 50, 2.0, 0.6, 8),
    (70, 55, 2.5, 0.6, 8),
    (75, 55, 3.0, 0.6, 8),
    (75, 60, 3.0, 0.8, 12),
    (70, 50, 1.5, 0.4, 6),
    (75, 50, 2.0, 0.4, 6),
    (80, 55, 2.5, 0.6, 12),
    (85, 60, 3.0, 0.8, 16),
    (75, 55, 3.0, 0.5, 8),
    (70, 55, 2.5, 0.5, 8),
    (80, 55, 2.5, 0.7, 10),
    (72, 52, 2.2, 0.55, 9),
    (75, 58, 2.8, 0.6, 10),
    (78, 58, 3.0, 0.6, 10),
    (80, 60, 3.5, 0.7, 12),
    (65, 45, 1.8, 0.5, 6),
    (70, 50, 2.0, 0.5, 8),
    (75, 55, 2.5, 0.7, 10),
    (80, 60, 3.0, 1.0, 12),
    (85, 65, 3.5, 1.0, 16),
]

for me, mc, tr, sa, cd in configs:
    tot_ret = 0
    tot_tr = 0
    tot_w = 0
    cnt = 0
    for sym, df in data.items():
        try:
            d = df.copy()
            d = calc_indicators(d)
            d = calc_signals(d, 20, mc, me)
            r = backtest_stock(d, tr, sa, cd)
            if r and r['trades'] > 0:
                tot_ret += r['return']
                tot_tr += r['trades']
                tot_w += r['trades'] * r['win_rate'] / 100
                cnt += 1
        except:
            pass
    if cnt >= 5 and tot_tr >= 20:
        results.append({
            'me': me, 'mc': mc, 'tr': tr, 'sa': sa, 'cd': cd,
            'ret': tot_ret/cnt, 'trades': tot_tr, 'win': tot_w/tot_tr*100 if tot_tr else 0,
            'n': cnt
        })

results.sort(key=lambda x: x['ret'], reverse=True)

print("="*70)
print("TOP PARAMETER COMBINATIONS (MICROCAP STOCKS)")
print("="*70)
for i, r in enumerate(results[:15]):
    print(f"{i+1}. exec>={r['me']} cand>={r['mc']} R={r['tr']} stop={r['sa']} cd={r['cd']}")
    print(f"   Return: {r['ret']:.1f}% | Trades: {r['trades']} | Win%: {r['win']:.1f}")

if results:
    best = results[0]
    print(f"\n=== OPTIMIZED PARAMETERS FOR STOCKS ===")
    print(f"i_min_exec = {best['me']}")
    print(f"i_min_candidate = {best['mc']}")
    print(f"i_target_r = {best['tr']}")
    print(f"i_stop_atr = {best['sa']}")
    print(f"i_cooldown = {best['cd']}")
    print(f"Expected Return: {best['ret']:.1f}%")
    print(f"Win Rate: {best['win']:.1f}%")
    
    # Show individual stock results with best params
    print(f"\n=== DETAILED RESULTS WITH BEST PARAMS ===")
    ME, MC, TR, SA, CD = best['me'], best['mc'], best['tr'], best['sa'], best['cd']
    for sym, df in data.items():
        d = df.copy()
        d = calc_indicators(d)
        d = calc_signals(d, 20, MC, ME)
        r = backtest_stock(d, TR, SA, CD)
        if r and r['trades'] > 0:
            print(f"{sym}: Trades={r['trades']}, Win%={r['win_rate']:.1f}%, Return={r['return']:.1f}%")