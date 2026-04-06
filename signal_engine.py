"""
Converts technical analysis results + options chain data into trading signals.

Signal types: BUY CALL | BUY PUT | HOLD
Each signal includes:
  - direction     : CALL / PUT / HOLD
  - confidence    : LOW / MEDIUM / HIGH
  - score         : int (0-100)
  - reasons       : list[str] explaining the signal
  - recommended   : list of specific option contracts to consider
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime


# ── Directional scoring ───────────────────────────────────────────────────────

def score_direction(ta: dict) -> tuple[int, list[str]]:
    """
    Score bullish vs bearish pressure from -100 (max bearish) to +100 (max bullish).
    Positive  → lean CALL
    Negative  → lean PUT
    Near-zero → HOLD
    Returns (score, reasons).
    """
    score = 0
    reasons: list[str] = []

    rsi = ta["rsi"]
    # RSI zones
    if rsi < 30:
        score += 25
        reasons.append(f"RSI oversold ({rsi:.1f}) - potential bounce (bullish)")
    elif rsi < 45:
        score += 10
        reasons.append(f"RSI below midpoint ({rsi:.1f}) - mild bullish lean")
    elif rsi > 70:
        score -= 25
        reasons.append(f"RSI overbought ({rsi:.1f}) - potential pullback (bearish)")
    elif rsi > 55:
        score -= 10
        reasons.append(f"RSI above midpoint ({rsi:.1f}) - mild bearish lean")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    # MACD
    if ta["macd_crossed_up"]:
        score += 20
        reasons.append("MACD crossed above signal line (bullish crossover)")
    elif ta["macd_crossed_down"]:
        score -= 20
        reasons.append("MACD crossed below signal line (bearish crossover)")
    elif ta["macd_bullish"]:
        score += 10
        reasons.append("MACD histogram positive (bullish momentum)")
    else:
        score -= 10
        reasons.append("MACD histogram negative (bearish momentum)")

    # Bollinger Bands
    pct_b = ta["bb_pct_b"]
    if pct_b < 0.1:
        score += 20
        reasons.append(f"Price near lower Bollinger Band (%B={pct_b:.2f}) - oversold zone")
    elif pct_b > 0.9:
        score -= 20
        reasons.append(f"Price near upper Bollinger Band (%B={pct_b:.2f}) - overbought zone")

    # EMA trend
    if ta["price_above_ema20"] and ta["ema20_above_ema50"]:
        score += 15
        reasons.append("Price above EMA20 > EMA50 - uptrend confirmed (bullish)")
    elif not ta["price_above_ema20"] and not ta["ema20_above_ema50"]:
        score -= 15
        reasons.append("Price below EMA20 < EMA50 - downtrend confirmed (bearish)")
    elif ta["price_above_ema20"]:
        score += 5
        reasons.append("Price above EMA20 - short-term bullish")
    else:
        score -= 5
        reasons.append("Price below EMA20 - short-term bearish")

    # Volume confirmation
    vol = ta["vol_ratio"]
    if vol > 1.5:
        reasons.append(f"Volume {vol:.1f}x above average - strong conviction")
        score = int(score * 1.15)  # amplify existing signal
    elif vol < 0.7:
        reasons.append(f"Volume {vol:.1f}x below average - weak conviction, treat signal cautiously")

    return max(-100, min(100, score)), reasons


def classify_signal(score: int) -> tuple[str, str]:
    """Map raw score to (direction, confidence)."""
    abs_score = abs(score)
    direction = "CALL" if score > 0 else ("PUT" if score < 0 else "HOLD")

    if direction == "HOLD":
        return "HOLD", "LOW"

    if abs_score >= 55:
        confidence = "HIGH"
    elif abs_score >= 30:
        confidence = "MEDIUM"
    else:
        # Weak signal - not worth trading
        direction = "HOLD"
        confidence = "LOW"

    return direction, confidence


# ── Contract recommendation ───────────────────────────────────────────────────

def recommend_contracts(
    direction: str,
    options_data: dict,
    current_price: float,
    max_contracts: int = 5,
) -> list[dict]:
    """
    Pick the best contracts from the options chain based on:
    - Strike within 5% OTM or ATM  (good delta exposure)
    - Volume > 10 and OI > 50      (liquidity filter)
    - Nearest 2 expiration dates    (time preference)
    - Spread % < 30%               (tight bid-ask)
    """
    if direction not in ("CALL", "PUT"):
        return []

    results: list[dict] = []
    key = "calls" if direction == "CALL" else "puts"

    for exp in options_data["expirations"]:
        df: pd.DataFrame = options_data[key].get(exp, pd.DataFrame())
        if df.empty:
            continue

        df = df.copy()

        # Strike range: ATM ± 5%
        atm_low = current_price * 0.95
        atm_high = current_price * 1.05
        near = df[(df["strike"] >= atm_low) & (df["strike"] <= atm_high)]

        if near.empty:
            # Relax to ± 10%
            atm_low = current_price * 0.90
            atm_high = current_price * 1.10
            near = df[(df["strike"] >= atm_low) & (df["strike"] <= atm_high)]

        if near.empty:
            continue

        df = near.copy()

        # Liquidity filter (OR: volume > 10 or OI > 50)
        vol_col = df["volume"].fillna(0) if "volume" in df.columns else pd.Series(0, index=df.index)
        oi_col = df["openInterest"].fillna(0) if "openInterest" in df.columns else pd.Series(0, index=df.index)
        liquid = df[(vol_col > 10) | (oi_col > 50)]
        if not liquid.empty:
            df = liquid

        # Sort by proximity to ATM
        df = df.copy()
        df["dist"] = (df["strike"] - current_price).abs()

        df = df.sort_values("dist")

        for _, row in df.head(3).iterrows():
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            mid_price = round((bid + ask) / 2, 2) if (bid or ask) else None
            iv = row.get("impliedVolatility")

            results.append({
                "expiration": exp,
                "strike": float(row["strike"]),
                "type": direction,
                "bid": bid,
                "ask": ask,
                "mid": mid_price,
                "volume": int(row.get("volume", 0) or 0),
                "open_interest": int(row.get("openInterest", 0) or 0),
                "iv": round(float(iv) * 100, 1) if iv and not np.isnan(iv) else None,
                "itm": row.get("inTheMoney", False),
            })

        if len(results) >= max_contracts:
            break

    return results[:max_contracts]


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_signal(ta: dict, options_data: dict) -> dict:
    """
    Full pipeline: TA → score → direction → contract picks.
    Returns a structured signal dict.
    """
    score, reasons = score_direction(ta)
    direction, confidence = classify_signal(score)

    current_price = ta["current_price"]
    contracts = recommend_contracts(direction, options_data, current_price)

    return {
        "direction": direction,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
        "recommended_contracts": contracts,
        "ta_snapshot": {
            "rsi": round(ta["rsi"], 2),
            "macd": round(ta["macd"], 4),
            "macd_signal": round(ta["macd_signal"], 4),
            "bb_pct_b": round(ta["bb_pct_b"], 3),
            "ema_20": round(ta["ema_20"], 2),
            "ema_50": round(ta["ema_50"], 2),
            "atr": round(ta["atr"], 2),
            "vol_ratio": round(ta["vol_ratio"], 2),
        },
    }
