"""
Options Trading Signal Bot
--------------------------
Usage:
    python main.py AAPL
    python main.py TSLA --expirations 4
    python main.py SPY --watch 60        # refresh every 60 seconds
"""

import argparse
import sys
import time

from rich.console import Console
from rich.progress import Progress, TextColumn

import data_fetcher
import technical_analysis
import signal_engine
import display

console = Console()


def run_analysis(ticker: str, num_expirations: int) -> None:
    ticker = ticker.upper().strip()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(f"Fetching data for [bold cyan]{ticker}[/bold cyan]...", total=None)

        # 1. Ticker info
        progress.update(task, description="Loading ticker info...")
        try:
            info = data_fetcher.get_ticker_info(ticker)
        except Exception as e:
            console.print(f"[red]Error loading ticker info: {e}[/red]")
            info = {"name": ticker, "current_price": None}

        # 2. Price history
        progress.update(task, description="Loading price history...")
        try:
            price_df = data_fetcher.get_stock_data(ticker)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return

        # 3. Options chain
        progress.update(task, description="Loading options chain...")
        try:
            options_data = data_fetcher.get_near_term_options(ticker, num_expirations)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return

        # 4. Technical analysis
        progress.update(task, description="Running technical analysis...")
        ta = technical_analysis.run_analysis(price_df)
        # Sync current price with latest TA value
        if info.get("current_price"):
            ta["current_price"] = info["current_price"]

        # 5. Generate signal
        progress.update(task, description="Generating signal...")
        signal = signal_engine.generate_signal(ta, options_data)

    # ── Output ──────────────────────────────────────────────────────────────
    console.print()
    display.print_ticker_header(ticker, info)
    console.print()
    display.print_ta_table(ta)
    console.print()
    display.print_signal(signal, ticker)
    console.print()
    display.print_contracts(signal["recommended_contracts"], signal["direction"])
    display.print_disclaimer()
    console.print()


def main():
    parser = argparse.ArgumentParser(
        prog="option-bot",
        description="Options trading signal generator — BUY CALL / BUY PUT / HOLD",
    )
    parser.add_argument("ticker", help="Stock/ETF ticker symbol (e.g. AAPL, SPY, TSLA)")
    parser.add_argument(
        "--expirations", "-e",
        type=int,
        default=3,
        help="Number of nearest expiration dates to analyze (default: 3)",
    )
    parser.add_argument(
        "--watch", "-w",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Auto-refresh every N seconds (0 = run once)",
    )

    args = parser.parse_args()

    if args.watch > 0:
        console.print(
            f"[bold cyan]Watch mode:[/bold cyan] refreshing every [bold]{args.watch}s[/bold]. "
            "Press Ctrl+C to stop.\n"
        )
        try:
            while True:
                console.clear()
                run_analysis(args.ticker, args.expirations)
                console.print(f"[dim]Next refresh in {args.watch}s...[/dim]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch mode stopped.[/yellow]")
    else:
        run_analysis(args.ticker, args.expirations)


if __name__ == "__main__":
    main()
