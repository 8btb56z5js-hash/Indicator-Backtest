'''Backtest for DCS v4.6 MODIFIED indicator using yfinance.
Implements core logic from the PineScript version: EMA crossovers, ATR adaptive bands, regime detection,
Bull/Bear scoring, entry/exit rules, and basic performance metrics.
'''

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Helper functions – EMA, ATR, etc.
# -----------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# -----------------------------------------------------------------------------
# Core DCS logic – translated from PineScript. Returns a DataFrame with the
# signals (approved_long, approved_short) and the supporting calculations.
# -----------------------------------------------------------------------------

def dcs_signals(df: pd.DataFrame, tf_seconds: int) -> pd.DataFrame:
    """Calculate DCS signals for a given timeframe.

    Parameters
    ----------
    df: DataFrame with columns ['Open','High','Low','Close','Volume']
    tf_seconds: length of the timeframe in seconds (used for auto‑adjust logic)

    Returns
    -------
    DataFrame with additional columns used for back‑testing, notably
    'approved_long' and 'approved_short'.
    """
    # ------- Auto‑timeframe adaptation ------------------------------------------------
    isUltraFastTf = tf_seconds <= 60
    isFastTf = tf_seconds <= 900
    isMidTf = tf_seconds > 900 and tf_seconds <= 3600

    # Parameter defaults from the script (these are the “auto” versions)
    pFast = 13 if isFastTf else (14 if isMidTf else 21)
    pMid = 21 if isFastTf else (34 if isMidTf else 55)
    pSlow = 55 if isFastTf else (89 if isMidTf else 144)
    pAtrLen = 14
    pVolLen = 20
    pSwing = 14 if isFastTf else (20 if isMidTf else 30)
    pErLen = 14 if isFastTf else (21 if isMidTf else 34)
    pCompLen = 60 if isFastTf else (75 if isMidTf else 100)

    # ------- Core indicators -----------------------------------------------------------
    df['emaFast'] = ema(df['Close'], pFast)
    df['emaMid'] = ema(df['Close'], pMid)
    df['emaSlow'] = ema(df['Close'], pSlow)
    df['atr'] = atr(df, pAtrLen)
    df['atrPct'] = df['atr'] / np.maximum(df['Close'], 0.000001)
    df['volAvg'] = df['Volume'].rolling(pVolLen).mean()

    # Adaptive band multiplier – mirrors the Pine logic
    df['bandMult'] = np.select(
        [df['atrPct'] > 0.08, df['atrPct'] > 0.04, df['atrPct'] > 0.02],
        [2.8, 2.4, 2.1],
        default=1.8,
    )
    df['upperBand'] = df['emaMid'] + df['atr'] * df['bandMult']
    df['lowerBand'] = df['emaMid'] - df['atr'] * df['bandMult']

    # Shifted values for previous bar calculations (as in PineScript)
    for col in ['Close','Open','High','Low','Volume','emaFast','emaMid','atr','atrPct','volAvg','upperBand','lowerBand']:
        df[f"{col}_1"] = df[col].shift(1)

    # ------- Higher‑timeframe context – simplified (use 1‑day TF for all) ----------
    # The original script requests a higher‑timeframe based on the current TF; for the
    # back‑test we approximate it with a daily EMA series and forward‑fill it.
    # This keeps the back‑test fast while preserving the spirit of the logic.
    if tf_seconds <= 86400:  # intraday – use daily EMA as HTF proxy
        daily = df.resample('1D').last()
        daily['htfFast'] = ema(daily['Close'], pFast)
        daily['htfMid'] = ema(daily['Close'], pMid)
        daily['htfSlow'] = ema(daily['Close'], pSlow)
        daily['htfClose'] = daily['Close']
        daily = daily[['htfFast','htfMid','htfSlow','htfClose']].ffill()
        df = df.join(daily, how='left')
        df[['htfFast','htfMid','htfSlow','htfClose']] = df[['htfFast','htfMid','htfSlow','htfClose']].ffill()
    else:
        df['htfFast'] = ema(df['Close'], pFast)
        df['htfMid'] = ema(df['Close'], pMid)
        df['htfSlow'] = ema(df['Close'], pSlow)
        df['htfClose'] = df['Close']

    df['htfBull'] = (df['htfClose'] > df['htfMid']) & (df['htfFast'] > df['htfMid']) & (df['htfMid'] >= df['htfSlow'])
    df['htfBear'] = (df['htfClose'] < df['htfMid']) & (df['htfFast'] < df['htfMid']) & (df['htfMid'] <= df['htfSlow'])
    df['htfNeutral'] = ~df['htfBull'] & ~df['htfBear']

    # ------- Market state features -----------------------------------------------------
    c = df['Close']
    c1 = df['Close_1']
    o1 = df['Open_1']
    h1 = df['High_1']
    l1 = df['Low_1']
    v1 = df['Volume_1']
    emaFast1 = df['emaFast_1']
    emaMid1 = df['emaMid_1']
    atr1 = df['atr_1']
    atrPct1 = df['atrPct_1']
    volAvg1 = df['volAvg_1']
    upperBand1 = df['upperBand_1']
    lowerBand1 = df['lowerBand_1']

    # Efficiency & compression (simplified version)
    directionalMove = (c1 - c.shift(pErLen + 1)).abs()
    pathMove = sum((c.shift(i) - c.shift(i+1)).abs() for i in range(1, pErLen+1))
    efficiency = np.where(pathMove > 0, directionalMove / pathMove, 0.0)

    # Trend detection
    bullTrend = (c1 > emaMid1) & (emaFast1 > emaMid1)
    bearTrend = (c1 < emaMid1) & (emaFast1 < emaMid1)
    trendStability = (efficiency > 0.34) & (bullTrend | bearTrend)

    # ----- Simple regime naming (focus on trend, compression, expansion) ------------
    regimeTrend = trendStability
    regimeCompression = (df['upperBand'] - df['lowerBand']).rolling(pCompLen).mean() > df['upperBand'].rolling(pCompLen).mean()
    regimeExpansion = ~regimeCompression & trendStability
    regimeName = np.where(regimeCompression, 'Compression',
                  np.where(regimeExpansion, 'Expansion',
                  np.where(regimeTrend, 'Trend', 'Mixed')))

    # ----- Scoring – the heavy part of the original script -----------------------
    # We'll compute a simplified version of the bull/bear score that captures the
    # main contributors: structure, momentum, displacement, HTF, liquidity, regime.
    # The exact numeric thresholds are kept close to the original.

    # Structure score
    bullStructure = np.where(bullTrend, 18, np.where(c1 > emaMid1, 10, 0))
    bearStructure = np.where(bearTrend, 18, np.where(c1 < emaMid1, 10, 0))

    # Momentum score
    momentumBull = ((c1 > o1) & (c1 > emaFast1) & (emaFast1 >= emaFast1.shift())).astype(int) * 14
    momentumBear = ((c1 < o1) & (c1 < emaFast1) & (emaFast1 <= emaFast1.shift())).astype(int) * 14

    # Displacement score – simplified based on body size & range
    body = (c1 - o1).abs()
    range_ = (h1 - l1).abs()
    bodyEff = np.where(range_ > 0, body / range_, 0.0)
    displacementBull = (bodyEff >= 0.55) & (range_ > atr1 * 0.80)
    displacementBear = displacementBull  # symmetrical for simplicity
    bullDisplacement = displacementBull.astype(int) * 10
    bearDisplacement = displacementBear.astype(int) * 10

    # HTF score – use the proxy we built above
    bullHtf = df['htfBull'].astype(int) * 14
    bearHtf = df['htfBear'].astype(int) * 14

    # Liquidity score – based on volume ratio
    volRatio = np.where(volAvg1 > 0, v1 / volAvg1, 1.0)
    bullLiquidity = np.where((volRatio >= 0.9) & (c1 > o1), 12, np.where(volRatio < 0.75, 0, 6))
    bearLiquidity = np.where((volRatio >= 0.9) & (c1 < o1), 12, np.where(volRatio < 0.75, 0, 6))

    # Regime score – simplified
    bullRegime = np.where(regimeTrend & bullTrend, 12, 0)
    bearRegime = np.where(regimeTrend & bearTrend, 12, 0)

    # Final scores
    df['bullScore'] = bullStructure + momentumBull + bullDisplacement + bullHtf + bullLiquidity + bullRegime
    df['bearScore'] = bearStructure + momentumBear + bearDisplacement + bearHtf + bearLiquidity + bearRegime

    # ----- Candidate detection -------------------------------------------------------
    # Pullback / breakout logic (simplified)
    swingHigh = df['High'].rolling(pSwing).max().shift(2)
    swingLow = df['Low'].rolling(pSwing).min().shift(2)
    pullbackLong = bullTrend & (l1 <= emaFast1 + atr1 * 0.35) & (c1 >= emaFast1) & (c1 > o1)
    pullbackShort = bearTrend & (h1 >= emaFast1 - atr1 * 0.35) & (c1 <= emaFast1) & (c1 < o1)
    breakoutLong = (c1 > swingHigh) & displacementBull & (volRatio >= 0.9)
    breakoutShort = (c1 < swingLow) & displacementBear & (volRatio >= 0.9)
    candidateLong = (pullbackLong | breakoutLong)
    candidateShort = (pullbackShort | breakoutShort)

    # ----- Execution scores ----------------------------------------------------------
    # Using the same minimum execution threshold as the script (i_min_exec = 75)
    i_min_exec = 75
    i_min_candidate = 58
    df['longExecutionScore'] = np.clip(df['bullScore'], 0, 100)
    df['shortExecutionScore'] = np.clip(df['bearScore'], 0, 100)

    df['approved_long'] = (
        candidateLong & (df['longExecutionScore'] >= i_min_exec)
    )
    df['approved_short'] = (
        candidateShort & (df['shortExecutionScore'] >= i_min_exec)
    )

    # Store some price levels for exit calculations
    df['entry_price'] = np.where(df['approved_long'], df['Close'], np.where(df['approved_short'], df['Close'], np.nan))
    df['target_price'] = np.nan
    df['stop_price'] = np.nan
    # Simplified target/stop: 1.8 * ATR risk as in the script
    i_target_r = 1.8
    i_stop_atr = 0.45
    df.loc[df['approved_long'], 'target_price'] = df['Close'] + i_target_r * atr1
    df.loc[df['approved_long'], 'stop_price'] = df['Close'] - i_stop_atr * atr1
    df.loc[df['approved_short'], 'target_price'] = df['Close'] - i_target_r * atr1
    df.loc[df['approved_short'], 'stop_price'] = df['Close'] + i_stop_atr * atr1

    return df

# -----------------------------------------------------------------------------
# Simple back‑test engine – walk through rows, open/close positions.
# -----------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame) -> dict:
    df = df.copy().reset_index()
    capital = 100_000  # starting equity
    position = None  # dict with keys: side, entry, target, stop, size
    trades = []
    for i, row in df.iterrows():
        # close position if stop/target hit before next bar open (simplified)
        if position:
            price = row['Close']
            # check stop/target on close price
            if position['side'] == 'long':
                if price <= position['stop'] or price >= position['target']:
                    pnl = (price - position['entry']) * position['size']
                    trades.append(pnl)
                    position = None
            else:  # short
                if price >= position['stop'] or price <= position['target']:
                    pnl = (position['entry'] - price) * position['size']
                    trades.append(pnl)
                    position = None
        # open new position if signal and no open position
        if not position:
            if row.get('approved_long'):
                risk = row['stop_price'] - row['entry_price']
                if risk <= 0:
                    continue
                size = capital * 0.02 / abs(risk)  # risk 2% per trade
                position = {
                    'side': 'long',
                    'entry': row['entry_price'],
                    'target': row['target_price'],
                    'stop': row['stop_price'],
                    'size': size,
                }
            elif row.get('approved_short'):
                risk = row['entry_price'] - row['stop_price']
                if risk <= 0:
                    continue
                size = capital * 0.02 / abs(risk)
                position = {
                    'side': 'short',
                    'entry': row['entry_price'],
                    'target': row['target_price'],
                    'stop': row['stop_price'],
                    'size': size,
                }
    # close any remaining open position at last close
    if position:
        price = df.iloc[-1]['Close']
        if position['side'] == 'long':
            pnl = (price - position['entry']) * position['size']
        else:
            pnl = (position['entry'] - price) * position['size']
        trades.append(pnl)
    # compute stats
    total_trades = len(trades)
    wins = [p for p in trades if p > 0]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    avg_r = np.mean([abs(p) / (capital * 0.02) for p in trades]) if trades else 0
    # max drawdown (simple equity curve)
    equity = capital
    equity_curve = []
    for p in trades:
        equity += p
        equity_curve.append(equity)
    peak = np.maximum.accumulate([capital] + equity_curve)
    drawdown = (np.array([capital] + equity_curve) - peak) / peak
    max_dd = drawdown.min() * 100 if len(drawdown) else 0
    return {
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'avg_R': round(avg_r, 2),
        'max_drawdown_pct': round(max_dd, 2),
    }

# -----------------------------------------------------------------------------
# Main routine – loop over symbols & timeframes
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
    # timeframes in minutes
    tf_map = {
        '1H': 60 * 60,
        '4H': 4 * 60 * 60,
        'D': 24 * 60 * 60,
    }
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=730)  # 24 months roughly
    for tf_label, tf_seconds in tf_map.items():
        print(f"\n=== Timeframe: {tf_label} ===")
        for sym in symbols:
            # download data
            df = yf.download(sym, start=start_date, end=end_date, interval=tf_label, progress=False)
            if df.empty:
                print(f"[WARN] No data for {sym} @ {tf_label}")
                continue
            df = df.rename(columns=lambda c: c.title())  # make titles match our code
            # compute signals
            df_sig = dcs_signals(df, tf_seconds)
            # run backtest
            stats = run_backtest(df_sig)
            print(f"{sym}: Trades={stats['total_trades']}, Win%={stats['win_rate']}, AvgR={stats['avg_R']}, MaxDD={stats['max_drawdown_pct']}%")
