"""
Options backtesting engine — vectorized, O(n) per ticker.

Strategy:
  - Precompute ALL indicators on the full DataFrame in one pass
  - Iterate bars only to manage open/close trade state
  - When BUY CALL fires: buy a 30-DTE ATM call at Black-Scholes price
  - When BUY PUT fires:  buy a 30-DTE ATM put  at Black-Scholes price

Exit rules (ATR-based stock price targets):
  - CALL Take Profit : stock rises 2.0 x ATR above entry price
  - CALL Stop Loss   : stock falls 1.0 x ATR below entry price
  - PUT  Take Profit : stock falls 2.0 x ATR below entry price
  - PUT  Stop Loss   : stock rises 1.0 x ATR above entry price
  - Hard floor       : option loses > 80% of premium → exit (decay protection)
  - Time Stop        : 5 DTE remaining → exit at market

Signal format:
  BUY CALL | AAPL | 2024-04-15 | Stock @ $165.20 | Strike $165 | Entry $4.35/share ($435/contract) | IV 28.4%
  SELL CALL | AAPL | 2024-05-10 | Stock @ $172.40 | Strike $165 | Exit $8.90/share ($890/contract) | P&L +$455 (+104.6%) | Take Profit (ATR)
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import options_pricer as pricer
import signal_filters as filters

warnings.filterwarnings("ignore")

# ── Strategy parameters ────────────────────────────────────────────────────────
WARMUP_BARS    = 55     # bars to prime indicators (EMA50 needs 50)
OPTION_DTE     = 30     # days to expiry at entry
TIME_STOP_DTE  = 5      # close when only N DTE remain
HARD_STOP_PCT  = -0.80  # exit if option loses 80% of entry value (decay floor)
ATR_TP_MULT    = 2.0    # take profit when stock moves this many ATR in our favor
ATR_SL_MULT    = 1.0    # stop loss when stock moves this many ATR against us
MIN_PREMIUM    = 0.05   # skip if option < $0.05
CONTRACTS      = 1      # 1 contract = 100 shares

USE_FILTERS    = True   # set False to disable all accuracy filters


# ── Vectorized indicator computation ──────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all TA indicators in one vectorized pass. Returns df + columns."""
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    out = df.copy()

    # RSI(14)
    delta    = close.diff()
    gain     = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss     = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    out["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD(12,26,9)
    ema12          = close.ewm(span=12, adjust=False).mean()
    ema26          = close.ewm(span=26, adjust=False).mean()
    macd           = ema12 - ema26
    sig            = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"]     = macd - sig
    out["macd_hist_lag"] = out["macd_hist"].shift(1)

    # Bollinger Bands %B (20, 2)
    sma20      = close.rolling(20).mean()
    std20      = close.rolling(20).std()
    bb_upper   = sma20 + 2 * std20
    bb_lower   = sma20 - 2 * std20
    out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # EMAs
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()

    # Historical volatility — 20-day rolling annualized
    log_ret    = np.log(close / close.shift(1))
    out["hv20"] = (log_ret.rolling(20).std() * np.sqrt(252)).clip(lower=0.05, upper=3.0)

    # Volume ratio vs 20-day average
    vol_avg         = vol.rolling(20).mean().replace(0, np.nan)
    out["vol_ratio"] = vol / vol_avg

    # ATR(14) — raw dollar value for exit targeting
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_val"] = tr.ewm(com=13, min_periods=14).mean()
    out["atr_pct"] = out["atr_val"] / close.replace(0, np.nan)

    # ADX(14) — vectorized
    up_move    = high.diff()
    down_move  = -low.diff()
    plus_dm    = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm   = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14      = tr.ewm(com=13, min_periods=14).mean()
    pdm        = pd.Series(plus_dm,  index=df.index).ewm(com=13, min_periods=14).mean()
    mdm        = pd.Series(minus_dm, index=df.index).ewm(com=13, min_periods=14).mean()
    pdi        = 100 * pdm / atr14.replace(0, np.nan)
    mdi        = 100 * mdm / atr14.replace(0, np.nan)
    dx         = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out["adx"] = dx.ewm(com=13, min_periods=14).mean()

    return out


# ── Directional score ─────────────────────────────────────────────────────────

def _score_row(row) -> int:
    """Score a single precomputed indicator row. Returns int in [-100, +100]."""
    score = 0
    rsi   = row["rsi"]
    if pd.isna(rsi):
        return 0

    # RSI
    if   rsi < 30:  score += 25
    elif rsi < 45:  score += 10
    elif rsi > 70:  score -= 25
    elif rsi > 55:  score -= 10

    # MACD histogram
    hist     = row["macd_hist"]
    hist_lag = row["macd_hist_lag"]
    if not (pd.isna(hist) or pd.isna(hist_lag)):
        if   hist > 0 and hist_lag <= 0: score += 20  # bullish crossover
        elif hist < 0 and hist_lag >= 0: score -= 20  # bearish crossover
        elif hist > 0:                   score += 10
        else:                            score -= 10

    # Bollinger %B
    pct_b = row["bb_pct_b"]
    if not pd.isna(pct_b):
        if   pct_b < 0.10: score += 20
        elif pct_b > 0.90: score -= 20

    # EMA trend
    ema20 = row["ema20"]
    ema50 = row["ema50"]
    price = row["Close"]
    if not (pd.isna(ema20) or pd.isna(ema50)):
        if   price > ema20 and ema20 > ema50: score += 15
        elif price < ema20 and ema20 < ema50: score -= 15
        elif price > ema20:                   score += 5
        else:                                 score -= 5

    # Volume amplifier
    vr = row["vol_ratio"]
    if not pd.isna(vr) and vr > 1.5:
        score = int(score * 1.15)

    return max(-100, min(100, score))


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:         int
    ticker:           str
    direction:        str
    entry_date:       object
    entry_stock:      float
    strike:           float
    expiry_date:      object
    entry_premium:    float
    entry_sigma:      float
    entry_atr:        float = 0.0
    tp_stock:         float = 0.0   # ATR-based take profit stock price
    sl_stock:         float = 0.0   # ATR-based stop loss stock price
    entry_indicators: dict  = field(default_factory=dict)
    exit_date:        object = None
    exit_stock:       float  = None
    exit_premium:     float  = None
    exit_reason:      str    = None

    @property
    def pnl_per_share(self):
        if self.exit_premium is None:
            return None
        return self.exit_premium - self.entry_premium

    @property
    def pnl_dollars(self):
        p = self.pnl_per_share
        return None if p is None else p * 100 * CONTRACTS

    @property
    def pnl_pct(self):
        if not self.entry_premium or self.pnl_per_share is None:
            return None
        return (self.pnl_per_share / self.entry_premium) * 100

    @property
    def is_win(self):
        return self.pnl_dollars is not None and self.pnl_dollars > 0

    def entry_line(self) -> str:
        return (
            f"BUY {self.direction:<4} | {self.ticker:<6} | {str(self.entry_date)[:10]} | "
            f"Stock @ ${self.entry_stock:>9.2f} | Strike ${self.strike:.2f} | "
            f"Entry ${self.entry_premium:.2f}/share (${self.entry_premium*100:.0f}/contract) | "
            f"IV {self.entry_sigma*100:.1f}% | ATR ${self.entry_atr:.2f} | Expiry {str(self.expiry_date)[:10]}"
        )

    def exit_line(self) -> str:
        if self.exit_date is None:
            return ""
        sign = "+" if (self.pnl_dollars or 0) >= 0 else ""
        return (
            f"SELL {self.direction:<4} | {self.ticker:<6} | {str(self.exit_date)[:10]} | "
            f"Stock @ ${self.exit_stock:>9.2f} | Strike ${self.strike:.2f} | "
            f"Exit  ${self.exit_premium:.2f}/share (${self.exit_premium*100:.0f}/contract) | "
            f"P&L {sign}${self.pnl_dollars:.0f} ({sign}{self.pnl_pct:.1f}%) | "
            f"{self.exit_reason}"
        )


# ── Main backtest loop ─────────────────────────────────────────────────────────

def backtest_ticker(
    ticker:           str,
    df:               pd.DataFrame,
    market_regime:    str              = "neutral",
    weekly_close:     pd.Series | None = None,
    earnings_dates:   set              = None,
    sector_regime_fn                   = None,   # callable(ticker, date) -> str
    ml_model                           = None,
) -> list[Trade]:
    """
    Vectorized options backtest for a single ticker.
    All filters and ATR-based exits are applied when USE_FILTERS=True.
    """
    if len(df) < WARMUP_BARS + 10:
        return []

    df = _compute_indicators(df.sort_index().copy())
    trades:     list[Trade] = []
    open_trade: Trade | None = None
    trade_id = 0

    rows = df.iloc[WARMUP_BARS:]

    for i, (ts, row) in enumerate(rows.iterrows()):
        price = float(row["Close"])
        if price <= 0 or pd.isna(price) or price > 50_000:
            continue

        # ── Manage open trade ────────────────────────────────────────────────
        if open_trade is not None:
            dte = max(0, (pd.Timestamp(open_trade.expiry_date) - ts).days)
            sigma = float(row["hv20"]) if not pd.isna(row.get("hv20", np.nan)) else open_trade.entry_sigma
            sigma = max(0.05, min(sigma, 3.0))

            # Price option at current stock price
            cur_prem = pricer.price_option(
                open_trade.direction, price, open_trade.strike, dte, sigma
            )
            pnl_pct = (cur_prem - open_trade.entry_premium) / open_trade.entry_premium

            exit_reason = None

            # ATR-based stock price exits (primary)
            if open_trade.direction == "CALL":
                if price >= open_trade.tp_stock:
                    exit_reason = "Take Profit (ATR)"
                elif price <= open_trade.sl_stock:
                    exit_reason = "Stop Loss (ATR)"
            else:  # PUT
                if price <= open_trade.tp_stock:
                    exit_reason = "Take Profit (ATR)"
                elif price >= open_trade.sl_stock:
                    exit_reason = "Stop Loss (ATR)"

            # Hard floor: option lost 80% of value (decay / wrong direction)
            if exit_reason is None and pnl_pct <= HARD_STOP_PCT:
                exit_reason = "Hard Stop (80% loss)"

            # Time stop
            if exit_reason is None and dte <= TIME_STOP_DTE:
                exit_reason = "Time Stop (5 DTE)"

            # End of data
            if exit_reason is None and i == len(rows) - 1:
                exit_reason = "End of Data"

            if exit_reason:
                open_trade.exit_date    = ts.date()
                open_trade.exit_stock   = round(price, 4)
                open_trade.exit_premium = round(cur_prem, 4)
                open_trade.exit_reason  = exit_reason
                trades.append(open_trade)
                open_trade = None
            continue

        # ── Check for new signal ─────────────────────────────────────────────
        score = _score_row(row)
        if abs(score) < 30:
            continue

        direction = "CALL" if score > 0 else "PUT"
        sigma = float(row["hv20"]) if not pd.isna(row.get("hv20", np.nan)) else 0.30
        sigma = max(0.05, min(sigma, 3.0))
        atr   = float(row["atr_val"]) if not pd.isna(row.get("atr_val", np.nan)) else price * 0.02
        atr   = max(atr, price * 0.005)   # floor at 0.5% of price

        # Build entry indicators dict (for ML and checkpoint storage)
        entry_ind = {
            "rsi":       float(row.get("rsi",       50)),
            "macd_hist": float(row.get("macd_hist",  0)),
            "bb_pct_b":  float(row.get("bb_pct_b",  0.5)),
            "ema20":     float(row.get("ema20",  price)),
            "ema50":     float(row.get("ema50",  price)),
            "hv20":      sigma,
            "adx":       float(row.get("adx",       20)),
            "vol_ratio": float(row.get("vol_ratio",  1)),
            "atr_pct":   atr / price if price > 0 else 0.02,
        }

        # ── Apply accuracy filters ───────────────────────────────────────────
        if USE_FILTERS:
            adx_val = float(row.get("adx", 25) or 25)

            # Sector regime at this date
            sector_reg = "neutral"
            if sector_regime_fn is not None:
                try:
                    sector_reg = sector_regime_fn(ticker, ts)
                except Exception:
                    pass

            passed, _ = filters.apply_all_filters(
                score          = score,
                direction      = direction,
                sigma          = sigma,
                adx_val        = adx_val,
                row            = row,
                regime         = market_regime,
                earnings_dates = earnings_dates or set(),
                sector_regime  = sector_reg,
                ml_model       = ml_model,
                entry_features = entry_ind,
            )
            if not passed:
                continue

        # ── Compute ATR-based stock price targets ────────────────────────────
        if direction == "CALL":
            tp_stock = price + ATR_TP_MULT * atr
            sl_stock = price - ATR_SL_MULT * atr
        else:
            tp_stock = price - ATR_TP_MULT * atr
            sl_stock = price + ATR_SL_MULT * atr

        strike  = pricer.atm_strike(price)
        expiry  = (ts + pd.Timedelta(days=OPTION_DTE)).date()
        premium = pricer.price_option(direction, price, strike, OPTION_DTE, sigma)

        if premium < MIN_PREMIUM or strike <= 0:
            continue

        trade_id += 1
        open_trade = Trade(
            trade_id         = trade_id,
            ticker           = ticker,
            direction        = direction,
            entry_date       = ts.date(),
            entry_stock      = round(price, 4),
            strike           = strike,
            expiry_date      = expiry,
            entry_premium    = round(premium, 4),
            entry_sigma      = round(sigma, 4),
            entry_atr        = round(atr, 4),
            tp_stock         = round(tp_stock, 4),
            sl_stock         = round(sl_stock, 4),
            entry_indicators = entry_ind,
        )

    return trades


# ── Aggregate stats ────────────────────────────────────────────────────────────

def compute_stats(trades: list[Trade]) -> dict:
    completed = [t for t in trades if t.exit_date is not None]
    if not completed:
        return {}

    wins   = [t for t in completed if t.is_win]
    losses = [t for t in completed if not t.is_win]
    pnls   = [t.pnl_dollars for t in completed]
    pcts   = [t.pnl_pct     for t in completed]

    win_rate = len(wins) / len(completed) * 100
    avg_win  = float(np.mean([t.pnl_dollars for t in wins]))   if wins   else 0.0
    avg_loss = float(np.mean([t.pnl_dollars for t in losses])) if losses else 0.0

    cum    = np.cumsum(pnls)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak))

    loss_total = sum(t.pnl_dollars for t in losses)
    win_total  = sum(t.pnl_dollars for t in wins)
    pf = abs(win_total / loss_total) if loss_total != 0 else float("inf")

    reasons = {}
    for t in completed:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    calls = [t for t in completed if t.direction == "CALL"]
    puts  = [t for t in completed if t.direction == "PUT"]

    return {
        "total_trades":  len(completed),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(win_rate, 1),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "avg_pnl":       round(float(np.mean(pnls)), 2),
        "total_pnl":     round(sum(pnls), 2),
        "profit_factor": round(pf, 2),
        "max_drawdown":  round(max_dd, 2),
        "avg_pnl_pct":   round(float(np.mean(pcts)), 1),
        "exit_reasons":  reasons,
        "call_trades":   len(calls),
        "put_trades":    len(puts),
    }
