"""
Black-Scholes options pricing for synthetic backtest P&L.
Uses historical realized volatility as a proxy for implied volatility.
"""

import numpy as np
from scipy.stats import norm


RISK_FREE_RATE = 0.045   # approximate 10-year treasury yield


def historical_volatility(close_series, window: int = 20) -> float:
    """
    Annualized historical volatility using log returns over `window` days.
    Returns a float (e.g. 0.30 = 30%).
    """
    if len(close_series) < window + 1:
        return 0.30  # fallback
    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    vol = float(log_returns.iloc[-window:].std() * np.sqrt(252))
    return max(0.05, min(vol, 3.0))   # clamp to [5%, 300%]


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    except (ZeroDivisionError, ValueError):
        return None, None


def bs_call(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes call price. T in years."""
    if T <= 0:
        return max(0.0, S - K)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(0.0, S - K)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(0.01, float(price))


def bs_put(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes put price. T in years."""
    if T <= 0:
        return max(0.0, K - S)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(0.0, K - S)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(0.01, float(price))


def atm_strike(price: float, increment: float = None) -> float:
    """Round price to nearest options strike increment."""
    if increment is None:
        if price < 10:
            increment = 0.50
        elif price < 25:
            increment = 1.0
        elif price < 50:
            increment = 1.0
        elif price < 200:
            increment = 2.5
        elif price < 500:
            increment = 5.0
        else:
            increment = 10.0
    return round(round(price / increment) * increment, 2)


def price_option(
    option_type: str,   # "CALL" or "PUT"
    stock_price: float,
    strike: float,
    days_to_expiry: int,
    sigma: float,
) -> float:
    """Price an option using Black-Scholes."""
    T = max(days_to_expiry, 0) / 365.0
    if option_type == "CALL":
        return bs_call(stock_price, strike, T, sigma)
    else:
        return bs_put(stock_price, strike, T, sigma)
