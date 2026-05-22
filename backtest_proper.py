#!/usr/bin/env python3
"""
PROPER Backtest for DCS v4.6 MODIFIED
Implements exact Pine Script signal logic
"""

import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np

SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
TIMEFRAMES = ['1h', '4h', '1d']
INITIAL_CAPITAL = 10000
RISK_PCT = 0.02

# Parameters (matching Pine Script)
P_FAST = 14
P_MID = 21
P_SLOW = 55
P_ATR = 14
P_VOL = 20
P_ER = 21
P_COMP = 60
P_SWING = 20
MIN_CANDIDATE = 58
MIN_EXEC = 75
TARGET_R = 1.8
STOP_ATR = 0.45
COOLDOWN = 8

def calculate_indicators(df):
    """Calculate all DCS indicators"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    open_ = df['Open']
    volume = df['Volume']
    
    # EMAs
    df['emaFast'] = close.ewm(span=P_FAST, adjust=False).mean()
    df['emaMid'] = close.ewm(span=P_MID, adjust=False).mean()
    df['emaSlow'] = close.ewm(span=P_SLOW, adjust=False).mean()
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(P_ATR).mean()
    
    # ATR bands
    df['atrPct'] = df['atr'] / close
    bandMult = np.select(
        [df['atrPct'] > 0.08, df['atrPct'] > 0.04, df['atrPct'] > 0.02],
        [2.8, 2.4, 2.1], default=1.8
    )
    df['upperBand'] = df['emaMid'] + df['atr'] * bandMult
    df['lowerBand'] = df['emaMid'] - df['atr'] * bandMult
    
    # Volume
    df['volAvg'] = volume.rolling(P_VOL).mean()
    
    return df

def calculate_features(df):
    """Calculate all features for signal generation"""
    c = df['Close'].shift(1)
    o = df['Open'].shift(1)
    h = df['High'].shift(1)
    l = df['Low'].shift(1)
    v = df['Volume'].shift(1)
    
    emaFast1 = df['emaFast'].shift(1)
    emaMid1 = df['emaMid'].shift(1)
    emaSlow1 = df['emaSlow'].shift(1)
    atr1 = df['atr'].shift(1)
    volAvg1 = df['volAvg'].shift(1)
    upperBand1 = df['upperBand'].shift(1)
    lowerBand1 = df['lowerBand'].shift(1)
    
    range1 = h - l
    body1 = (c - o).abs()
    bodyEff1 = body1 / range1.replace(0, np.nan)
    upperWick1 = h - pd.concat([o, c], axis=1).max(axis=1)
    lowerWick1 = pd.concat([o, c], axis=1).min(axis=1) - l
    
    # Previous 2 bars for swing detection
    h2 = df['High'].shift(2)
    l2 = df['Low'].shift(2)
    swingHigh = h2.rolling(P_SWING).max()
    swingLow = l2.rolling(P_SWING).min()
    
    # Volume ratio
    volRatio = v / volAvg1.replace(0, np.nan)
    
    # Trend detection
    bullTrend = (c > emaMid1) & (emaFast1 > emaMid1)
    bearTrend = (c < emaMid1) & (emaFast1 < emaMid1)
    
    # Momentum
    momentumBull = (c > o) & (c > emaFast1) & (emaFast1 >= df['emaFast'].shift(2))
    momentumBear = (c < o) & (c < emaFast1) & (emaFast1 <= df['emaFast'].shift(2))
    
    # Displacement
    displacementBull = (c > o) & (bodyEff1 >= 0.55) & (range1 > atr1 * 0.80)
    displacementBear = (c < o) & (bodyEff1 >= 0.55) & (range1 > atr1 * 0.80)
    
    # Participation
    thinParticipation = volRatio < 0.75
    qualityParticipation = volRatio >= 0.90
    
    # Setup patterns
    pullbackLong = bullTrend & (l <= emaFast1 + atr1 * 0.35) & (c >= emaFast1) & (c > o)
    pullbackShort = bearTrend & (h >= emaFast1 - atr1 * 0.35) & (c <= emaFast1) & (c < o)
    breakoutLong = (c > swingHigh) & displacementBull & qualityParticipation
    breakoutShort = (c < swingLow) & displacementBear & qualityParticipation
    reclaimLong = (c > emaMid1) & (df['Close'].shift(2) <= df['emaMid'].shift(2)) & displacementBull
    rejectShort = (c < emaMid1) & (df['Close'].shift(2) >= df['emaMid'].shift(2)) & displacementBear
    
    # Candidate raw signals
    candidateLongRaw = pullbackLong | breakoutLong | reclaimLong
    candidateShortRaw = pullbackShort | breakoutShort | rejectShort
    
    # Bull/Bear Scoring
    bullStructure = np.where(bullTrend, 18, np.where(c > emaMid1, 10, 0))
    bearStructure = np.where(bearTrend, 18, np.where(c < emaMid1, 10, 0))
    bullMomentum = np.where(momentumBull, 14, 0)
    bearMomentum = np.where(momentumBear, 14, 0)
    bullDisplacement = np.where(displacementBull, 10, 0)
    bearDisplacement = np.where(displacementBear, 10, 0)
    
    # Simplified HTF (using prior bars as proxy)
    htfBull = bullTrend
    htfBear = bearTrend
    bullHtf = np.where(htfBull, 14, 0)
    bearHtf = np.where(htfBear, 14, 0)
    
    # Liquidity
    bullLiquidity = np.where(qualityParticipation & (c > o), 12, np.where(thinParticipation, 0, 6))
    bearLiquidity = np.where(qualityParticipation & (c < o), 12, np.where(thinParticipation, 0, 6))
    
    # Regime (simplified)
    regimeTrend = bullTrend | bearTrend
    bullRegime = np.where(regimeTrend & bullTrend, 12, 5)
    bearRegime = np.where(regimeTrend & bearTrend, 12, 5)
    
    # Total scores
    bullScore = bullStructure + bullMomentum + bullDisplacement + bullHtf + bullLiquidity + bullRegime
    bearScore = bearStructure + bearMomentum + bearDisplacement + bearHtf + bearLiquidity + bearRegime
    
    # Bias
    biasLong = bullScore > (bearScore + 8)
    biasShort = bearScore > (bullScore + 8)
    
    # Entry quality
    longEntryQuality = np.where(pullbackLong, 22, np.where(breakoutLong, 18, 6))
    shortEntryQuality = np.where(pullbackShort, 22, np.where(breakoutShort, 18, 6))
    
    # Candidate scores
    longCandidateScore = np.where(bullScore > 0, bullScore + longEntryQuality, 0)
    shortCandidateScore = np.where(bearScore > 0, bearScore + shortEntryQuality, 0)
    
    # Risk checks
    riskParticipationLong = thinParticipation
    riskParticipationShort = thinParticipation
    riskVolatility = (df['atrPct'] > 0.028) & (~qualityParticipation)
    riskHtfLong = bearTrend & (~breakoutLong)
    riskHtfShort = bullTrend & (~breakoutShort)
    
    # Risk block
    longBlock = riskParticipationLong | riskVolatility | riskHtfLong
    shortBlock = riskParticipationShort | riskVolatility | riskHtfShort
    
    # Execution score (simplified)
    longExecScore = np.where(longBlock, 0, longCandidateScore - 20)
    shortExecScore = np.where(shortBlock, 0, shortCandidateScore - 20)
    
    # Approved signals
    approvedLong = candidateLongRaw & biasLong & (longCandidateScore >= MIN_CANDIDATE) & (longExecScore >= MIN_EXEC)
    approvedShort = candidateShortRaw & biasShort & (shortCandidateScore >= MIN_CANDIDATE) & (shortExecScore >= MIN_EXEC)
    
    # Store in dataframe
    df['approvedLong'] = approvedLong
    df['approvedShort'] = approvedShort
    df['swingHigh'] = swingHigh
    df['swingLow'] = swingLow
    df['atr1'] = atr1
    
    return df

def run_backtest(df):
    """Run backtest with proper position management"""
    df = df.dropna()
    if len(df) < 50:
        return None
    
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    cooldown_bars = 0
    
    for i in range(50, len(df)):
        # Cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1
        
        price = df['Close'].iloc[i]
        atr = df['atr1'].iloc[i]
        
        if position is None and cooldown_bars == 0:
            # Entry signals
            if df['approvedLong'].iloc[i]:
                stop = price - atr * STOP_ATR
                target = price + (price - stop) * TARGET_R
                risk = price - stop
                size = (capital * RISK_PCT) / risk
                position = {
                    'type': 'long',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'size': size
                }
                cooldown_bars = COOLDOWN
            elif df['approvedShort'].iloc[i]:
                stop = price + atr * STOP_ATR
                target = price - (stop - price) * TARGET_R
                risk = stop - price
                size = (capital * RISK_PCT) / risk
                position = {
                    'type': 'short',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'size': size
                }
                cooldown_bars = COOLDOWN
        elif position is not None:
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
                position = None
    
    if len(trades) == 0:
        return None
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    risk_reward = INITIAL_CAP_CAP = INITIAL_CAPITAL * RISK_PCT
    
    avg_R = np.mean([t['pnl'] / risk_reward for t in trades]) if trades else 0
    
    # Max drawdown
    equity = [INITIAL_CAPITAL]
    for t in trades:
        equity.append(equity[-1] + t['pnl'])
    
    max_dd = 0
    peak = equity[0]
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_R': avg_R,
        'max_dd': max_dd,
        'final_capital': capital,
        'return_pct': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    }

print("="*60)
print("DCS v4.6 PROPER Backtest")
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
            df = calculate_features(df)
            
            result = run_backtest(df)
            if result and result['trades'] > 0:
                results.append(result)
                print(f"{sym}: Trades={result['trades']}, Win%={result['win_rate']:.1f}%, AvgR={result['avg_R']:.2f}, MaxDD={result['max_dd']:.1f}%, Return={result['return_pct']:.1f}%")
            else:
                print(f"{sym}: No signals")
        except Exception as e:
            print(f"{sym}: Error - {str(e)[:50]}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if results:
    total_trades = sum(r['trades'] for r in results)
    avg_win = np.mean([r['win_rate'] for r in results])
    avg_return = np.mean([r['return_pct'] for r in results])
    avg_dd = np.mean([r['max_dd'] for r in results])
    avg_r = np.mean([r['avg_R'] for r in results])
    print(f"Total Trades: {total_trades}")
    print(f"Avg Win Rate: {avg_win:.1f}%")
    print(f"Avg Return: {avg_return:.1f}%")
    print(f"Avg Avg R: {avg_r:.2f}")
    print(f"Avg Max DD: {avg_dd:.1f}%")
    print(f"Instruments tested: {len(results)}")