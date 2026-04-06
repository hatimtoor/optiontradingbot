"""
Options backtesting system — with checkpoint/resume support.

If the run is interrupted, simply re-run the same command and it will
skip already-completed tickers and continue from where it left off.

Usage:
    python run_backtest.py                   # download + run full backtest
    python run_backtest.py --no-download     # skip download, use cached data
    python run_backtest.py --ticker AAPL     # run only for one ticker
    python run_backtest.py --log             # show full trade log after summary
    python run_backtest.py --best-worst      # show best/worst individual trades
    python run_backtest.py --reset           # clear checkpoints and start over
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)

import data_downloader
import backtester
import backtest_report
from tickers import TICKERS

warnings.filterwarnings("ignore")

console = Console()

RESULTS_FILE    = Path("backtest_results.json")
CHECKPOINT_DIR  = Path("backtest_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _checkpoint_path(ticker: str) -> Path:
    return CHECKPOINT_DIR / f"{ticker}.pkl"


def _save_checkpoint(ticker: str, trades: list, stats: dict) -> None:
    with open(_checkpoint_path(ticker), "wb") as f:
        pickle.dump({"trades": trades, "stats": stats}, f)


def _load_checkpoint(ticker: str):
    p = _checkpoint_path(ticker)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _completed_tickers() -> set[str]:
    return {p.stem for p in CHECKPOINT_DIR.glob("*.pkl")}


def _reset_checkpoints() -> None:
    for p in CHECKPOINT_DIR.glob("*.pkl"):
        p.unlink()
    console.print("[yellow]Checkpoints cleared. Starting fresh.[/yellow]\n")


# ── Main run ───────────────────────────────────────────────────────────────────

def run(args):
    if args.reset:
        _reset_checkpoints()

    # ── Step 1: Download / locate data ──────────────────────────────────────
    disk_mb = data_downloader._disk_mb()

    if args.no_download or (disk_mb > 10 and not args.force_download):
        console.print(f"[dim]Using cached data ({disk_mb:.0f} MB on disk). "
                      f"Pass --force-download to refresh.[/dim]\n")
        ticker_list = data_downloader.available_tickers()
    else:
        download_stats = data_downloader.download_all(TICKERS, size_limit_gb=1.8)
        console.print(
            f"\n[green]Download complete:[/green] "
            f"{download_stats['tickers_ok']} tickers, "
            f"{download_stats['total_daily_bars']:,} daily bars, "
            f"{download_stats['total_hourly_bars']:,} hourly bars, "
            f"{download_stats['disk_mb']:.0f} MB on disk\n"
        )
        ticker_list = data_downloader.available_tickers()

    if args.ticker:
        ticker_list = [t for t in ticker_list if t.upper() == args.ticker.upper()]

    if not ticker_list:
        console.print("[red]No data found. Run without --no-download first.[/red]")
        return

    # ── Step 2: Compute market regime from SPY (used for all tickers) ──────────
    import signal_filters as sf
    spy_df = data_downloader.load_daily("SPY")
    market_regime = sf.compute_market_regime(spy_df)
    console.print(f"[dim]Market regime (SPY): [bold]{market_regime.upper()}[/bold][/dim]\n")

    # ── Step 3: Identify already-done tickers ────────────────────────────────
    done = _completed_tickers()
    remaining = [t for t in ticker_list if t not in done]

    if done and not args.ticker:
        console.print(
            f"[dim]Resuming: {len(done)} tickers already done, "
            f"{len(remaining)} remaining.[/dim]\n"
        )
    else:
        console.print(f"[bold cyan]Running backtest on {len(ticker_list)} tickers...[/bold cyan]\n")

    # ── Step 3: Backtest remaining tickers ───────────────────────────────────
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

                    # Build weekly close for multi-TF filter
                    weekly_close = None
                    try:
                        weekly_close = df["Close"].resample("W").last().dropna()
                    except Exception:
                        pass

                    trades = backtester.backtest_ticker(
                        ticker, df,
                        market_regime=market_regime,
                        weekly_close=weekly_close,
                    )
                    stats  = backtester.compute_stats(trades)
                    _save_checkpoint(ticker, trades, stats)

                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Interrupted! Progress saved. "
                        "Re-run to continue from here.[/yellow]"
                    )
                    return
                except Exception as e:
                    _save_checkpoint(ticker, [], {})

                progress.advance(task)

    # ── Step 4: Load all checkpoints ─────────────────────────────────────────
    all_trades: list = []
    all_stats:  dict = {}

    for ticker in ticker_list:
        ck = _load_checkpoint(ticker)
        if ck is None:
            continue
        all_trades.extend(ck["trades"])
        if ck["stats"]:
            all_stats[ticker] = ck["stats"]

    completed_trades = [t for t in all_trades if t.exit_date is not None]
    if not completed_trades:
        console.print("[red]No completed trades found. Check your data.[/red]")
        return

    # ── Step 5: Save consolidated JSON results ───────────────────────────────
    results_data = {
        "tickers_analyzed": len(all_stats),
        "total_trades": len(completed_trades),
        "disk_mb": data_downloader._disk_mb(),
        "per_ticker": all_stats,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    # ── Step 6: Display results ──────────────────────────────────────────────
    console.print()
    backtest_report.print_overall_summary(all_stats, len(all_stats), data_downloader._disk_mb())
    console.print()
    backtest_report.print_ticker_stats(all_stats)
    console.print()
    backtest_report.print_exit_breakdown(all_trades)

    if args.log:
        ticker_filter = args.ticker.upper() if args.ticker else None
        backtest_report.print_trade_log(
            all_trades, max_trades=100, filter_ticker=ticker_filter
        )

    if args.best_worst:
        backtest_report.print_best_worst(all_trades, n=10)

    # Always show a sample of recent trades
    console.print("\n[bold]Sample Signals (10 most recent closed trades):[/bold]\n")
    recent = sorted(completed_trades, key=lambda t: t.entry_date, reverse=True)[:10]
    for t in recent:
        d_style = "green" if t.direction == "CALL" else "red"
        p_style = "green" if t.is_win else "red"
        console.print(f"  [{d_style}]{t.entry_line()}[/{d_style}]")
        console.print(f"  [{p_style}]{t.exit_line()}[/{p_style}]")
        console.print()

    console.print(f"[dim]Full results saved to {RESULTS_FILE}[/dim]")
    console.print(f"[dim]Checkpoints in {CHECKPOINT_DIR}/ — run with --reset to start over[/dim]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="run_backtest",
        description="Options signal backtester with checkpoint/resume support",
    )
    parser.add_argument("--no-download",    action="store_true",
                        help="Skip download, use cached data on disk")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download all tickers even if cached")
    parser.add_argument("--ticker",  "-t",  type=str, default=None,
                        help="Run for a single ticker only")
    parser.add_argument("--log",            action="store_true",
                        help="Print full trade log after summary")
    parser.add_argument("--best-worst",     action="store_true",
                        help="Show top 10 best and worst individual trades")
    parser.add_argument("--reset",          action="store_true",
                        help="Clear all checkpoints and start the backtest over")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
