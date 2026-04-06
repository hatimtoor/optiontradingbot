"""
Downloads and caches historical OHLCV data for all tickers.
Targets up to ~1.5 GB of raw data stored as compressed Parquet files.

Layout:
  data/daily/<TICKER>.parquet   — full history, daily bars
  data/hourly/<TICKER>.parquet  — last 2 years, 1-hour bars
"""

import os
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.progress import (
    Progress, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)

warnings.filterwarnings("ignore")

console = Console()

DATA_DIR   = Path("data")
DAILY_DIR  = DATA_DIR / "daily"
HOURLY_DIR = DATA_DIR / "hourly"

DAILY_DIR.mkdir(parents=True, exist_ok=True)
HOURLY_DIR.mkdir(parents=True, exist_ok=True)


def _save(df: pd.DataFrame, path: Path) -> None:
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(path, compression="snappy", index=True)


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _disk_mb() -> float:
    total = 0
    for f in DATA_DIR.rglob("*.parquet"):
        total += f.stat().st_size
    return total / (1024 ** 2)


def download_ticker(ticker: str, force: bool = False) -> dict:
    """
    Download both daily (max history) and hourly (2y) data for one ticker.
    Returns dict with keys 'daily_rows', 'hourly_rows', 'skipped'.
    """
    daily_path  = DAILY_DIR  / f"{ticker}.parquet"
    hourly_path = HOURLY_DIR / f"{ticker}.parquet"
    result = {"daily_rows": 0, "hourly_rows": 0, "skipped": False}

    # ── Daily ──────────────────────────────────────────────────────────────
    if not force and daily_path.exists():
        result["daily_rows"] = len(_load(daily_path))
    else:
        try:
            df = yf.download(ticker, period="max", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                return result
            # Flatten multi-level columns from yfinance ≥0.2.x
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            _save(df, daily_path)
            result["daily_rows"] = len(df)
        except Exception:
            pass

    # ── Hourly ─────────────────────────────────────────────────────────────
    if not force and hourly_path.exists():
        result["hourly_rows"] = len(_load(hourly_path))
    else:
        try:
            df = yf.download(ticker, period="2y", interval="1h",
                             progress=False, auto_adjust=True)
            if df.empty:
                return result
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            _save(df, hourly_path)
            result["hourly_rows"] = len(df)
        except Exception:
            pass

    return result


def download_all(tickers: list[str], size_limit_gb: float = 1.8) -> dict:
    """
    Download all tickers, stopping if total disk usage approaches the limit.
    Returns summary stats.
    """
    total_daily = 0
    total_hourly = 0
    downloaded = 0
    skipped = 0
    errors = 0

    console.print(f"\n[bold cyan]Downloading data for {len(tickers)} tickers[/bold cyan]  "
                  f"[dim](limit: {size_limit_gb} GB)[/dim]\n")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(tickers))

        for ticker in tickers:
            used_mb = _disk_mb()
            if used_mb >= size_limit_gb * 1024:
                console.print(f"[yellow]Size limit reached ({used_mb:.0f} MB). Stopping.[/yellow]")
                break

            progress.update(task, description=f"[cyan]{ticker:<8}[/cyan] | {used_mb:.0f} MB on disk")

            result = download_ticker(ticker)
            total_daily  += result["daily_rows"]
            total_hourly += result["hourly_rows"]

            if result["daily_rows"] > 0 or result["hourly_rows"] > 0:
                downloaded += 1
            else:
                errors += 1

            progress.advance(task)
            # Small delay to avoid rate limiting
            time.sleep(0.2)

    final_mb = _disk_mb()
    return {
        "tickers_ok": downloaded,
        "tickers_failed": errors,
        "total_daily_bars": total_daily,
        "total_hourly_bars": total_hourly,
        "disk_mb": final_mb,
    }


def load_daily(ticker: str) -> pd.DataFrame | None:
    path = DAILY_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = _load(path)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df
    except Exception:
        return None


def load_hourly(ticker: str) -> pd.DataFrame | None:
    path = HOURLY_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = _load(path)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df
    except Exception:
        return None


def available_tickers() -> list[str]:
    """Return list of tickers that have daily data on disk."""
    return sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
