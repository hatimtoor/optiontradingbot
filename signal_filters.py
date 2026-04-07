"""
Accuracy filters applied BEFORE entering a trade.
Each filter returns (pass: bool, reason: str).

Filters implemented:
  1. IV Cap            — skip when IV > 35% (high IV = expensive premiums, negative return)
  2. ADX Strength      — skip extremely flat/ranging markets
  3. Market Regime     — don't trade against SPY macro trend
  4. Earnings Filter   — skip within 5 days of earnings (IV crush kills buyers)
  5. Sector Alignment  — require sector ETF to agree with trade direction
  6. ML Filter         — skip if GradientBoosting model predicts < 52% win chance
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ── Tuneable parameters ────────────────────────────────────────────────────────
IV_CAP            = 0.35    # reject if IV > 35%  (key insight: >35% IV = negative avg return)
ADX_MIN           = 15.0    # loose trend filter — only skip extremely flat/ranging markets
SCORE_THRESHOLD   = 30      # same as original; IV cap is the real quality gate
CONFLUENCE_MIN    = 3       # minimum indicators agreeing (only used if explicitly called)


# ── 1. IV Cap ─────────────────────────────────────────────────────────────────

def iv_cap_filter(sigma: float) -> tuple[bool, str]:
    """Skip when historical/implied volatility is too high."""
    if sigma > IV_CAP:
        return False, f"IV too high ({sigma*100:.1f}% > {IV_CAP*100:.0f}% cap) - premium is expensive"
    return True, ""


# ── 2. ADX Trend Strength ─────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute Average Directional Index (ADX) for the most recent bar.
    ADX > 20 = trending market; ADX < 20 = choppy/ranging (avoid trading).
    """
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)

    if len(df) < period * 2:
        return 25.0  # assume trending if insufficient data

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr       = pd.Series(plus_dm, index=df.index).ewm(com=period-1, min_periods=period).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(com=period-1, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(com=period-1, min_periods=period).mean() / atr.replace(0, np.nan)

    dx_num   = (plus_di - minus_di).abs()
    dx_den   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * dx_num / dx_den
    adx      = dx.ewm(com=period-1, min_periods=period).mean()

    val = float(adx.iloc[-1])
    return val if not np.isnan(val) else 25.0


def adx_filter(adx_value: float) -> tuple[bool, str]:
    """Skip when ADX is below minimum — market is choppy, not trending."""
    if adx_value < ADX_MIN:
        return False, f"ADX too low ({adx_value:.1f} < {ADX_MIN}) - ranging/choppy market"
    return True, ""


# ── 3. Score Threshold ────────────────────────────────────────────────────────

def score_threshold_filter(score: int) -> tuple[bool, str]:
    """Require higher conviction before entering."""
    if abs(score) < SCORE_THRESHOLD:
        return False, f"Score {score:+d} below threshold (|score| < {SCORE_THRESHOLD})"
    return True, ""


# ── 4. Confluence Counter ─────────────────────────────────────────────────────

def confluence_filter(row: pd.Series, direction: str) -> tuple[bool, str]:
    """
    Count how many individual indicators independently agree with the direction.
    Requires at least CONFLUENCE_MIN agreements.

    Indicators counted:
      - RSI: bullish if < 45, bearish if > 55
      - MACD histogram: bullish if > 0
      - BB %B: bullish if < 0.35, bearish if > 0.65
      - Price vs EMA20: bullish if above, bearish if below
      - EMA20 vs EMA50: bullish if EMA20 > EMA50, bearish if below
    """
    bull_signals = 0
    bear_signals = 0

    rsi   = row.get("rsi",   np.nan)
    hist  = row.get("macd_hist", np.nan)
    pct_b = row.get("bb_pct_b",  np.nan)
    price = row.get("Close", np.nan)
    ema20 = row.get("ema20", np.nan)
    ema50 = row.get("ema50", np.nan)

    if not np.isnan(rsi):
        if rsi < 45:  bull_signals += 1
        elif rsi > 55: bear_signals += 1

    if not np.isnan(hist):
        if hist > 0: bull_signals += 1
        else:        bear_signals += 1

    if not np.isnan(pct_b):
        if pct_b < 0.35: bull_signals += 1
        elif pct_b > 0.65: bear_signals += 1

    if not (np.isnan(price) or np.isnan(ema20)):
        if price > ema20: bull_signals += 1
        else:             bear_signals += 1

    if not (np.isnan(ema20) or np.isnan(ema50)):
        if ema20 > ema50: bull_signals += 1
        else:             bear_signals += 1

    agreement = bull_signals if direction == "CALL" else bear_signals
    total     = bull_signals + bear_signals

    if agreement < CONFLUENCE_MIN:
        return False, (
            f"Low confluence ({agreement}/{total} indicators agree for {direction}) "
            f"- need {CONFLUENCE_MIN}+"
        )
    return True, ""


# ── 5. Market Regime ──────────────────────────────────────────────────────────

def compute_market_regime(spy_df: pd.DataFrame | None) -> str:
    """
    Determine broad market regime from SPY.
    Returns: 'bull', 'bear', or 'neutral'
    Uses 50-day EMA slope and distance from 200-day EMA.
    """
    if spy_df is None or len(spy_df) < 200:
        return "neutral"

    close  = spy_df["Close"].astype(float)
    ema50  = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    price_last = float(close.iloc[-1])
    ema50_last = float(ema50.iloc[-1])
    ema200_last = float(ema200.iloc[-1])

    # EMA50 slope over last 5 days
    ema50_slope = (ema50.iloc[-1] - ema50.iloc[-6]) / ema50.iloc[-6]

    above_200 = price_last > ema200_last
    ema50_rising = ema50_slope > 0.002   # at least 0.2% rise

    if above_200 and ema50_rising:
        return "bull"
    elif not above_200 and not ema50_rising:
        return "bear"
    return "neutral"


def regime_direction_filter(regime: str, direction: str) -> tuple[bool, str]:
    """
    In a strong bull regime, avoid PUT signals (fight the trend).
    In a strong bear regime, avoid CALL signals.
    In neutral, allow both.
    """
    if regime == "bull" and direction == "PUT":
        return False, "Market regime is BULL - skipping PUT signal (fighting trend)"
    if regime == "bear" and direction == "CALL":
        return False, "Market regime is BEAR - skipping CALL signal (fighting trend)"
    return True, ""


# ── 6. Multi-Timeframe EMA ────────────────────────────────────────────────────

def multi_tf_ema_filter(
    daily_row: pd.Series,
    weekly_close: pd.Series | None,
    direction: str,
) -> tuple[bool, str]:
    """
    Check that the weekly EMA20 trend agrees with the daily signal direction.
    If weekly is falling and we want to BUY CALL, skip it.
    """
    if weekly_close is None or len(weekly_close) < 20:
        return True, ""   # insufficient weekly data — don't block

    w_ema20 = float(weekly_close.ewm(span=20, adjust=False).mean().iloc[-1])
    w_ema10 = float(weekly_close.ewm(span=10, adjust=False).mean().iloc[-1])
    w_price = float(weekly_close.iloc[-1])

    weekly_bullish = w_price > w_ema20 and w_ema10 > w_ema20
    weekly_bearish = w_price < w_ema20 and w_ema10 < w_ema20

    if direction == "CALL" and weekly_bearish:
        return False, "Weekly trend is BEARISH - daily CALL signal conflicts with weekly trend"
    if direction == "PUT" and weekly_bullish:
        return False, "Weekly trend is BULLISH - daily PUT signal conflicts with weekly trend"

    return True, ""


# ── 4. Earnings Filter ────────────────────────────────────────────────────────

EARNINGS_WINDOW = 5   # days — skip signals this close to earnings

def earnings_filter(
    signal_date,
    earnings_dates: set,
) -> tuple[bool, str]:
    """
    Skip signals within EARNINGS_WINDOW days of a known earnings date.
    Earnings events cause IV crush that destroys long option premium value.
    earnings_dates: set of datetime.date objects for this ticker.
    """
    if not earnings_dates:
        return True, ""

    import datetime
    if hasattr(signal_date, "date"):
        signal_date = signal_date.date()

    for ed in earnings_dates:
        if hasattr(ed, "date"):
            ed = ed.date()
        try:
            delta = abs((signal_date - ed).days)
            if delta <= EARNINGS_WINDOW:
                return False, f"Earnings within {delta} days ({ed}) - IV crush risk"
        except Exception:
            continue

    return True, ""


# ── 5. Sector ETF Alignment ────────────────────────────────────────────────────

# Map each ticker to its primary sector ETF
SECTOR_ETF_MAP: dict[str, str] = {
    # Technology / Software / Semis
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK", "GOOG": "XLK",
    "META": "XLK", "AMD":  "XLK", "INTC": "XLK", "QCOM":  "XLK", "MU":   "XLK",
    "AMAT": "XLK", "LRCX": "XLK", "KLAC": "XLK", "TXN":   "XLK", "AVGO": "XLK",
    "MRVL": "XLK", "SMCI": "XLK", "ARM":  "XLK", "TSM":   "XLK",
    "ORCL": "XLK", "CRM":  "XLK", "ADBE": "XLK", "CSCO":  "XLK", "IBM":  "XLK",
    "SNOW": "XLK", "DDOG": "XLK", "CRWD": "XLK", "NET":   "XLK", "PANW": "XLK",
    "ZS":   "XLK", "OKTA": "XLK", "NOW":  "XLK", "WDAY":  "XLK", "VEEV": "XLK",
    # Communication / Media / Internet
    "NFLX": "XLK", "SNAP": "XLK", "PINS": "XLK", "RDDT":  "XLK",
    "DIS":  "XLK", "CMCSA":"XLK", "T":    "XLK", "VZ":    "XLK", "TMUS": "XLK",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "SHOP": "XLY", "NKE":   "XLY", "LULU": "XLY",
    "SBUX": "XLY", "MCD":  "XLY", "YUM":  "XLY", "CMG":   "XLY", "HD":   "XLY",
    "LOW":  "XLY", "TGT":  "XLY", "ETSY": "XLY", "UBER":  "XLY", "LYFT": "XLY",
    "ABNB": "XLY", "DASH": "XLY", "RBLX": "XLY",
    "F":    "XLY", "GM":   "XLY", "RIVN": "XLY", "LCID":  "XLY",
    "NIO":  "XLY", "XPEV": "XLY", "LI":   "XLY",
    # Financials
    "JPM":  "XLF", "BAC":  "XLF", "C":    "XLF", "WFC":   "XLF", "GS":   "XLF",
    "MS":   "XLF", "BLK":  "XLF", "V":    "XLF", "MA":    "XLF", "AXP":  "XLF",
    "PYPL": "XLF", "COIN": "XLF", "SOFI": "XLF", "HOOD":  "XLF",
    "MET":  "XLF", "PRU":  "XLF", "AFL":  "XLF",
    # Healthcare / Biotech / Pharma
    "JNJ":  "XLV", "PFE":  "XLV", "MRNA": "XLV", "BNTX":  "XLV", "ABBV": "XLV",
    "UNH":  "XLV", "CVS":  "XLV", "LLY":  "XLV", "BMY":   "XLV", "MRK":  "XLV",
    "AMGN": "XLV", "GILD": "XLV", "BIIB": "XLV", "REGN":  "XLV",
    "ISRG": "XLV", "DHR":  "XLV", "BSX":  "XLV",
    # Energy
    "XOM":  "XLE", "CVX":  "XLE", "COP":  "XLE", "OXY":   "XLE",
    "SLB":  "XLE", "HAL":  "XLE", "MPC":  "XLE", "VLO":   "XLE", "USO":  "XLE",
    # Industrials / Aerospace / Defense
    "BA":   "XLI", "GE":   "XLI", "CAT":  "XLI", "DE":    "XLI", "MMM":  "XLI",
    "HON":  "XLI", "LMT":  "XLI", "RTX":  "XLI", "NOC":   "XLI", "GD":   "XLI",
    # Consumer Staples
    "WMT":  "XLP", "COST": "XLP",
    # Real Estate
    "AMT":  "XLRE", "PLD": "XLRE", "SPG":  "XLRE", "O":    "XLRE",
    # Materials / Commodities (use XLB or GLD as proxy)
    "GDX":  "XLB", "GDXJ":"XLB",
    "GLD":  "XLB", "IAU":  "XLB", "SLV":  "XLB",
    # China → FXI
    "BABA": "FXI", "JD":   "FXI", "FXI":  "FXI",
}


def sector_alignment_filter(
    direction:     str,
    sector_regime: str,
) -> tuple[bool, str]:
    """
    Only take a signal if the sector ETF trend agrees with the trade direction.
    - BUY CALL in a bearish sector = fighting the sector → skip
    - BUY PUT in a bullish sector = fighting the sector → skip
    'neutral' sectors always pass.
    """
    if direction == "CALL" and sector_regime == "bear":
        return False, "Sector ETF is BEARISH - skipping CALL signal"
    if direction == "PUT" and sector_regime == "bull":
        return False, "Sector ETF is BULLISH - skipping PUT signal"
    return True, ""


def compute_sector_regimes(df: pd.DataFrame) -> pd.Series:
    """
    For a sector ETF's price DataFrame, compute a daily regime series.
    Returns pd.Series indexed by date with values 'bull', 'bear', 'neutral'.
    """
    if df is None or len(df) < 55:
        return pd.Series(dtype=str)

    close  = df["Close"].astype(float)
    ema20  = close.ewm(span=20, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()

    bull = (close > ema50) & (ema20 > ema50)
    bear = (close < ema50) & (ema20 < ema50)

    regime = pd.Series("neutral", index=close.index)
    regime[bull] = "bull"
    regime[bear] = "bear"
    return regime


# ── Master filter pipeline ────────────────────────────────────────────────────

def apply_all_filters(
    score:          int,
    direction:      str,
    sigma:          float,
    adx_val:        float,
    row:            pd.Series,
    regime:         str              = "neutral",
    weekly_close:   pd.Series | None = None,
    earnings_dates: set              = None,
    sector_regime:  str              = "neutral",
    ml_model                         = None,
    entry_features: dict | None      = None,
) -> tuple[bool, list[str]]:
    """
    Run all filters in sequence (fail-fast).
    Returns (should_trade, list_of_rejection_reasons).
    """
    import ml_signal

    checks = [
        iv_cap_filter(sigma),
        adx_filter(adx_val),
        regime_direction_filter(regime, direction),
        sector_alignment_filter(direction, sector_regime),
        earnings_filter(
            row.name if hasattr(row, "name") else None,
            earnings_dates or set(),
        ),
        ml_signal.ml_filter(ml_model, entry_features or {}),
    ]

    rejections = []
    for passed, reason in checks:
        if not passed:
            rejections.append(reason)
            return False, rejections   # fail-fast

    return True, []
