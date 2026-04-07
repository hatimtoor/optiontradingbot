"""
Options backtesting system — checkpoint/resume + all accuracy filters.

Improvements over baseline:
  1. Dynamic ATR-based exits (2x ATR take profit, 1x ATR stop loss)
  2. Earnings filter (skip within 5 days of earnings)
  3. Sector ETF alignment (trade in direction of sector trend)
  4. ML filter (GradientBoosting win probability prediction)

Workflow:
  First run  : python run_backtest.py --no-download
  Train ML   : python ml_signal.py --train
  Second run : python run_backtest.py --no-download   (now uses ML filter too)

Usage:
    python run_backtest.py                    # download + full backtest
    python run_backtest.py --no-download      # use cached data
    python run_backtest.py --ticker AAPL      # single ticker
    python run_backtest.py --log              # show full trade log
    python run_backtest.py --best-worst       # show best/worst trades
    python run_backtest.py --reset            # clear checkpoints, start over
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

import data_downloader
import backtester
import backtest_report
import signal_filters as sf
import ml_signal
from tickers import TICKERS

warnings.filterwarnings("ignore")

console = Console()

RESULTS_FILE      = Path("backtest_results.json")
CHECKPOINT_DIR    = Path("backtest_checkpoints")
EARNINGS_CACHE    = Path("data/earnings_dates.json")
SECTOR_CACHE      = Path("data/sector_regimes.pkl")

CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _save_checkpoint(ticker: str, trades: list, stats: dict) -> None:
    path = CHECKPOINT_DIR / f"{ticker}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"trades": trades, "stats": stats}, f)


def _load_checkpoint(ticker: str):
    path = CHECKPOINT_DIR / f"{ticker}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _completed_tickers() -> set[str]:
    return {p.stem for p in CHECKPOINT_DIR.glob("*.pkl")}


def _reset_checkpoints() -> None:
    for p in CHECKPOINT_DIR.glob("*.pkl"):
        p.unlink()
    console.print("[yellow]Checkpoints cleared. Starting fresh.[/yellow]\n")


# ── Earnings dates ─────────────────────────────────────────────────────────────

def _download_earnings_dates(tickers: list[str]) -> dict[str, list[str]]:
    """Download and cache earnings dates for all tickers."""
    console.print("[cyan]Downloading earnings dates (cached after first run)...[/cyan]")
    result: dict[str, list[str]] = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.earnings_dates
            if df is not None and not df.empty:
                dates = [str(d.date()) for d in df.index if not pd.isna(d)]
                result[ticker] = dates
            else:
                result[ticker] = []
            time.sleep(0.1)
        except Exception:
            result[ticker] = []
    return result


def load_earnings_dates(tickers: list[str]) -> dict[str, set]:
    """Returns {ticker: set_of_date_objects}."""
    import datetime

    if EARNINGS_CACHE.exists():
        try:
            with open(EARNINGS_CACHE) as f:
                raw = json.load(f)
        except Exception:
            raw = {}
    else:
        raw = _download_earnings_dates(tickers)
        EARNINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(EARNINGS_CACHE, "w") as f:
            json.dump(raw, f)

    # Convert to sets of date objects
    result = {}
    for ticker, date_strs in raw.items():
        dates = set()
        for ds in date_strs:
            try:
                dates.add(datetime.date.fromisoformat(str(ds)[:10]))
            except Exception:
                pass
        result[ticker] = dates
    return result


# ── Sector regime precomputation ───────────────────────────────────────────────

def build_sector_regimes() -> dict[str, pd.Series]:
    """
    For each unique sector ETF in SECTOR_ETF_MAP, load its daily data
    and compute a date-indexed Series of 'bull'/'bear'/'neutral'.
    Returns {etf_ticker: pd.Series}.
    """
    if SECTOR_CACHE.exists():
        try:
            with open(SECTOR_CACHE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    unique_etfs = set(sf.SECTOR_ETF_MAP.values())
    console.print(f"[cyan]Computing sector regimes for {len(unique_etfs)} ETFs...[/cyan]")

    regimes: dict[str, pd.Series] = {}
    for etf in sorted(unique_etfs):
        df = data_downloader.load_daily(etf)
        if df is not None and len(df) >= 55:
            regimes[etf] = sf.compute_sector_regimes(df)
        else:
            regimes[etf] = pd.Series(dtype=str)

    with open(SECTOR_CACHE, "wb") as f:
        pickle.dump(regimes, f)
    return regimes


def make_sector_regime_fn(regimes: dict[str, pd.Series]):
    """Returns a function: (ticker, date) -> 'bull'/'bear'/'neutral'."""
    def get_regime(ticker: str, date: pd.Timestamp) -> str:
        etf = sf.SECTOR_ETF_MAP.get(ticker.upper())
        if etf is None:
            return "neutral"
        series = regimes.get(etf)
        if series is None or series.empty:
            return "neutral"
        # Normalize timezone
        try:
            idx = series.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
                series = pd.Series(series.values, index=idx)
            past = series[idx <= date]
            return past.iloc[-1] if not past.empty else "neutral"
        except Exception:
            return "neutral"
    return get_regime


# ── Main run ───────────────────────────────────────────────────────────────────

def run(args):
    if args.reset:
        _reset_checkpoints()

    # ── Step 1: Data ────────────────────────────────────────────────────────
    disk_mb = data_downloader._disk_mb()

    if args.no_download or (disk_mb > 10 and not args.force_download):
        console.print(f"[dim]Using cached data ({disk_mb:.0f} MB). Pass --force-download to refresh.[/dim]\n")
        ticker_list = data_downloader.available_tickers()
    else:
        stats = data_downloader.download_all(TICKERS, size_limit_gb=1.8)
        console.print(
            f"\n[green]Download complete:[/green] {stats['tickers_ok']} tickers, "
            f"{stats['total_daily_bars']:,} daily bars, {stats['disk_mb']:.0f} MB\n"
        )
        ticker_list = data_downloader.available_tickers()

    if args.ticker:
        ticker_list = [t for t in ticker_list if t.upper() == args.ticker.upper()]

    if not ticker_list:
        console.print("[red]No data found. Run without --no-download first.[/red]")
        return

    # ── Step 2: Market regime (SPY) ─────────────────────────────────────────
    spy_df = data_downloader.load_daily("SPY")
    market_regime = sf.compute_market_regime(spy_df)
    console.print(f"[dim]Market regime (SPY): [bold]{market_regime.upper()}[/bold][/dim]")

    # ── Step 3: Sector regimes ───────────────────────────────────────────────
    sector_regimes  = build_sector_regimes()
    sector_regime_fn = make_sector_regime_fn(sector_regimes)

    # ── Step 4: Earnings dates ───────────────────────────────────────────────
    earnings_map = load_earnings_dates(ticker_list)
    console.print(
        f"[dim]Earnings dates loaded for {sum(1 for v in earnings_map.values() if v)} tickers.[/dim]\n"
    )

    # ── Step 5: ML model ─────────────────────────────────────────────────────
    model = ml_signal.load_model()
    if model:
        console.print("[dim]ML model loaded — applying win probability filter.[/dim]\n")
    else:
        console.print(
            "[dim]No ML model found. Run 'python ml_signal.py --train' after first backtest "
            "to enable the ML filter.[/dim]\n"
        )

    # ── Step 6: Backtest remaining tickers ───────────────────────────────────
    done      = _completed_tickers()
    remaining = [t for t in ticker_list if t not in done]

    if done and not args.ticker:
        console.print(
            f"[dim]Resuming: {len(done)} done, {len(remaining)} remaining. "
            f"Use --reset to start over.[/dim]\n"
        )
    else:
        console.print(f"[bold cyan]Running backtest on {len(ticker_list)} tickers...[/bold cyan]\n")

    if remaining:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Backtesting...", total=len(remaining))

            for ticker in remaining:
                progress.update(task, description=f"[cyan]{ticker:<8}[/cyan]")
                try:
                    df = data_downloader.load_daily(ticker)
                    if df is None or len(df) < 60:
                        _save_checkpoint(ticker, [], {})
                        progress.advance(task)
                        continue

                    # Weekly close for optional multi-TF
                    weekly_close = None
                    try:
                        weekly_close = df["Close"].resample("W").last().dropna()
                    except Exception:
                        pass

                    trades = backtester.backtest_ticker(
                        ticker           = ticker,
                        df               = df,
                        market_regime    = market_regime,
                        weekly_close     = weekly_close,
                        earnings_dates   = earnings_map.get(ticker, set()),
                        sector_regime_fn = sector_regime_fn,
                        ml_model         = model,
                    )
                    stats = backtester.compute_stats(trades)
                    _save_checkpoint(ticker, trades, stats)

                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Interrupted — progress saved. Re-run to continue.[/yellow]"
                    )
                    return
                except Exception:
                    _save_checkpoint(ticker, [], {})

                progress.advance(task)

    # ── Step 7: Load all checkpoints ─────────────────────────────────────────
    all_trades: list = []
    all_stats:  dict = {}

    for ticker in ticker_list:
        ck = _load_checkpoint(ticker)
        if ck is None:
            continue
        all_trades.extend(ck["trades"])
        if ck["stats"]:
            all_stats[ticker] = ck["stats"]

    completed = [t for t in all_trades if t.exit_date is not None]
    if not completed:
        console.print("[red]No completed trades found.[/red]")
        return

    # ── Step 8: Save JSON results ─────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(
            {"tickers_analyzed": len(all_stats), "total_trades": len(completed),
             "disk_mb": disk_mb, "per_ticker": all_stats},
            f, indent=2, default=str,
        )

    # ── Step 9: Display ───────────────────────────────────────────────────────
    console.print()
    backtest_report.print_overall_summary(all_stats, len(all_stats), disk_mb)
    console.print()
    backtest_report.print_ticker_stats(all_stats)
    console.print()
    backtest_report.print_exit_breakdown(all_trades)

    if args.log:
        backtest_report.print_trade_log(
            all_trades, max_trades=100,
            filter_ticker=args.ticker.upper() if args.ticker else None,
        )

    if args.best_worst:
        backtest_report.print_best_worst(all_trades, n=10)

    # Quick sample
    console.print("\n[bold]Sample Signals (10 most recent closed trades):[/bold]\n")
    recent = sorted(completed, key=lambda t: t.entry_date, reverse=True)[:10]
    for t in recent:
        d_style = "green" if t.direction == "CALL" else "red"
        p_style = "green" if t.is_win else "red"
        console.print(f"  [{d_style}]{t.entry_line()}[/{d_style}]")
        console.print(f"  [{p_style}]{t.exit_line()}[/{p_style}]")
        console.print()

    console.print(f"[dim]Results: {RESULTS_FILE} | Checkpoints: {CHECKPOINT_DIR}/[/dim]")
    if not model:
        console.print(
            "[bold yellow]Next step: python ml_signal.py --train  "
            "then re-run to add ML filter.[/bold yellow]"
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Options signal backtester")
    parser.add_argument("--no-download",    action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--ticker",  "-t",  type=str, default=None)
    parser.add_argument("--log",            action="store_true")
    parser.add_argument("--best-worst",     action="store_true")
    parser.add_argument("--reset",          action="store_true",
                        help="Clear checkpoints and re-run everything")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
