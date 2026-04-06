"""
Technical indicators used to generate directional bias for options signals.
All calculations operate on a pandas DataFrame with OHLCV columns.
"""

import numpy as np
import pandas as pd


# ── RSI ────────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ───────────────────────────────────────────────────────────────────────

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Bollinger Bands ────────────────────────────────────────────────────────────

def compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    # %B: where price sits within the band (0 = lower, 1 = upper)
    pct_b = (close - lower) / (upper - lower)
    return upper, sma, lower, pct_b


# ── EMA trend ─────────────────────────────────────────────────────────────────

def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


# ── ATR (volatility proxy) ─────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ── Volume trend ──────────────────────────────────────────────────────────────

def volume_trend(df: pd.DataFrame, period: int = 20) -> float:
    """
    Returns ratio of latest volume vs its rolling average.
    > 1.5  → high volume confirmation
    < 0.7  → low conviction
    """
    avg_vol = df["Volume"].rolling(period).mean().iloc[-1]
    if avg_vol == 0 or np.isnan(avg_vol):
        return 1.0
    return float(df["Volume"].iloc[-1] / avg_vol)


# ── Aggregate into a structured result ────────────────────────────────────────

def run_analysis(df: pd.DataFrame) -> dict:
    """
    Run all indicators on price DataFrame and return the latest values.
    """
    close = df["Close"]

    rsi = compute_rsi(close)
    macd_line, signal_line, histogram = compute_macd(close)
    upper_bb, mid_bb, lower_bb, pct_b = compute_bollinger(close)
    ema_20 = compute_ema(close, 20)
    ema_50 = compute_ema(close, 50)
    atr = compute_atr(df)

    latest = close.iloc[-1]

    return {
        "current_price": float(latest),
        "rsi": float(rsi.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_histogram": float(histogram.iloc[-1]),
        "bb_upper": float(upper_bb.iloc[-1]),
        "bb_mid": float(mid_bb.iloc[-1]),
        "bb_lower": float(lower_bb.iloc[-1]),
        "bb_pct_b": float(pct_b.iloc[-1]),
        "ema_20": float(ema_20.iloc[-1]),
        "ema_50": float(ema_50.iloc[-1]),
        "atr": float(atr.iloc[-1]),
        "vol_ratio": volume_trend(df),
        # Derived flags
        "price_above_ema20": latest > ema_20.iloc[-1],
        "price_above_ema50": latest > ema_50.iloc[-1],
        "ema20_above_ema50": ema_20.iloc[-1] > ema_50.iloc[-1],
        "macd_bullish": histogram.iloc[-1] > 0,
        "macd_crossed_up": (histogram.iloc[-1] > 0) and (histogram.iloc[-2] <= 0),
        "macd_crossed_down": (histogram.iloc[-1] < 0) and (histogram.iloc[-2] >= 0),
    }
