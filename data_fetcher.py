"""
Fetches stock price history and options chain data using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


def get_stock_data(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV price history for a ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. Check the symbol.")
    df.index = pd.to_datetime(df.index)
    return df


def get_options_chain(ticker: str) -> dict:
    """
    Fetch all available options expirations and chain data.
    Returns a dict with:
      - expirations: list of expiry date strings
      - calls: dict[expiry] -> DataFrame
      - puts:  dict[expiry] -> DataFrame
      - current_price: float
    """
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise ValueError(f"No options data available for '{ticker}'.")

    current_price = _get_current_price(stock)

    calls_by_exp = {}
    puts_by_exp = {}

    for exp in expirations:
        chain = stock.option_chain(exp)
        calls_by_exp[exp] = chain.calls
        puts_by_exp[exp] = chain.puts

    return {
        "expirations": list(expirations),
        "calls": calls_by_exp,
        "puts": puts_by_exp,
        "current_price": current_price,
    }


def get_near_term_options(ticker: str, num_expirations: int = 3) -> dict:
    """
    Return only the nearest N expiration dates to focus on near-term signals.
    """
    data = get_options_chain(ticker)
    near_exps = data["expirations"][:num_expirations]
    return {
        "expirations": near_exps,
        "calls": {e: data["calls"][e] for e in near_exps},
        "puts": {e: data["puts"][e] for e in near_exps},
        "current_price": data["current_price"],
    }


def get_ticker_info(ticker: str) -> dict:
    """Return basic info about the ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    return {
        "name": info.get("shortName") or info.get("longName") or ticker.upper(),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap"),
        "current_price": _get_current_price(stock),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "beta": info.get("beta"),
    }


def _get_current_price(stock: yf.Ticker) -> float:
    """Extract the most recent price from a Ticker object."""
    try:
        price = stock.fast_info.get("lastPrice") or stock.fast_info.get("regularMarketPrice")
        if price:
            return float(price)
    except Exception:
        pass
    hist = stock.history(period="1d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    raise ValueError("Could not retrieve current price.")
