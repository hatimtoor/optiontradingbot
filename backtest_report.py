"""
Rich terminal output for backtest results.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def _pnl_style(val: float) -> str:
    return "bold green" if val > 0 else ("bold red" if val < 0 else "dim")


def _fmt_pnl(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.0f}"


def _fmt_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


# ── Summary banner ─────────────────────────────────────────────────────────────

def print_overall_summary(all_stats: dict, total_tickers: int, disk_mb: float) -> None:
    total_trades = sum(s.get("total_trades", 0) for s in all_stats.values())
    total_wins   = sum(s.get("wins", 0) for s in all_stats.values())
    total_pnl    = sum(s.get("total_pnl", 0) for s in all_stats.values())
    avg_win_rate = (total_wins / total_trades * 100) if total_trades else 0
    avg_pnl_trade = total_pnl / total_trades if total_trades else 0

    all_pnl_pcts = [s.get("avg_pnl_pct", 0) for s in all_stats.values() if s]
    avg_pnl_pct  = sum(all_pnl_pcts) / len(all_pnl_pcts) if all_pnl_pcts else 0

    pnl_style = _pnl_style(total_pnl)

    lines = [
        f"  Tickers analyzed : [bold white]{total_tickers}[/bold white]",
        f"  Data on disk     : [bold white]{disk_mb:.0f} MB[/bold white]",
        f"  Total trades     : [bold white]{total_trades:,}[/bold white]",
        f"  Overall win rate : [bold white]{avg_win_rate:.1f}%[/bold white]",
        f"  Avg P&L per trade: [{pnl_style}]{_fmt_pnl(avg_pnl_trade)}[/{pnl_style}]",
        f"  Avg return/trade : [{pnl_style}]{_fmt_pct(avg_pnl_pct)}[/{pnl_style}]",
        f"  Total P&L (sim)  : [{pnl_style}]{_fmt_pnl(total_pnl)}[/{pnl_style}]",
    ]
    console.print(Panel("\n".join(lines), title="[bold]Backtest Overall Summary[/bold]",
                        border_style="cyan"))


# ── Per-ticker stats table ────────────────────────────────────────────────────

def print_ticker_stats(all_stats: dict) -> None:
    table = Table(
        title="Per-Ticker Backtest Results",
        box=box.ROUNDED,
        border_style="blue",
        show_lines=False,
    )
    table.add_column("Ticker",    style="cyan bold", width=8)
    table.add_column("Trades",    justify="right", width=7)
    table.add_column("Wins",      justify="right", width=6)
    table.add_column("Win %",     justify="right", width=7)
    table.add_column("Avg P&L",   justify="right", width=10)
    table.add_column("Avg Ret%",  justify="right", width=9)
    table.add_column("Total P&L", justify="right", width=12)
    table.add_column("Max DD",    justify="right", width=11)
    table.add_column("PF",        justify="right", width=6)
    table.add_column("Calls",     justify="right", width=6)
    table.add_column("Puts",      justify="right", width=5)

    # Sort by total P&L descending
    sorted_tickers = sorted(all_stats.items(), key=lambda x: x[1].get("total_pnl", 0), reverse=True)

    for ticker, s in sorted_tickers:
        if not s:
            continue
        pnl_style = _pnl_style(s["total_pnl"])
        wr_style  = "green" if s["win_rate"] >= 50 else "red"
        table.add_row(
            ticker,
            str(s["total_trades"]),
            str(s["wins"]),
            f"[{wr_style}]{s['win_rate']}%[/{wr_style}]",
            f"[{_pnl_style(s['avg_pnl'])}]{_fmt_pnl(s['avg_pnl'])}[/{_pnl_style(s['avg_pnl'])}]",
            f"[{_pnl_style(s['avg_pnl_pct'])}]{_fmt_pct(s['avg_pnl_pct'])}[/{_pnl_style(s['avg_pnl_pct'])}]",
            f"[{pnl_style}]{_fmt_pnl(s['total_pnl'])}[/{pnl_style}]",
            f"[red]{_fmt_pnl(s['max_drawdown'])}[/red]",
            str(s["profit_factor"]),
            str(s["call_trades"]),
            str(s["put_trades"]),
        )

    console.print(table)


# ── Trade log ─────────────────────────────────────────────────────────────────

def print_trade_log(trades: list, max_trades: int = 50, filter_ticker: str = None) -> None:
    """Print individual BUY/SELL signal lines for each trade."""
    filtered = [t for t in trades if filter_ticker is None or t.ticker == filter_ticker]
    filtered = [t for t in filtered if t.exit_date is not None]
    filtered = sorted(filtered, key=lambda t: t.entry_date)

    if not filtered:
        console.print("[yellow]No completed trades to display.[/yellow]")
        return

    # Show a representative sample — top 25 by abs P&L + 25 most recent
    by_abs = sorted(filtered, key=lambda t: abs(t.pnl_dollars), reverse=True)[:max_trades // 2]
    by_recent = filtered[-(max_trades // 2):]
    combined = list({t.trade_id: t for t in by_abs + by_recent}.values())
    combined.sort(key=lambda t: t.entry_date)

    console.print(f"\n[bold]Trade Log[/bold] [dim](showing {len(combined)} of {len(filtered)} trades)[/dim]\n")

    for t in combined:
        pnl_style = "green" if t.is_win else "red"
        direction_style = "green" if t.direction == "CALL" else "red"
        console.print(f"  [{direction_style}]{t.entry_line()}[/{direction_style}]")
        if t.exit_date:
            console.print(f"  [{pnl_style}]{t.exit_line()}[/{pnl_style}]")
        console.print()


# ── Exit reason breakdown ─────────────────────────────────────────────────────

def print_exit_breakdown(all_trades: list) -> None:
    reasons: dict[str, int] = {}
    for t in all_trades:
        if t.exit_reason:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    total = sum(reasons.values())
    if not total:
        return

    table = Table(title="Exit Reason Breakdown", box=box.SIMPLE, border_style="dim")
    table.add_column("Reason", style="white")
    table.add_column("Count", justify="right")
    table.add_column("% of Trades", justify="right")

    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        table.add_row(reason, str(count), f"{count/total*100:.1f}%")

    console.print(table)


# ── Best / Worst trades ────────────────────────────────────────────────────────

def print_best_worst(all_trades: list, n: int = 10) -> None:
    completed = [t for t in all_trades if t.exit_date is not None]
    if not completed:
        return

    best  = sorted(completed, key=lambda t: t.pnl_dollars, reverse=True)[:n]
    worst = sorted(completed, key=lambda t: t.pnl_dollars)[:n]

    console.print(f"\n[bold green]Top {n} Winning Trades:[/bold green]")
    for t in best:
        console.print(f"  [green]{t.entry_line()}[/green]")
        console.print(f"  [green]{t.exit_line()}[/green]\n")

    console.print(f"\n[bold red]Top {n} Losing Trades:[/bold red]")
    for t in worst:
        console.print(f"  [red]{t.entry_line()}[/red]")
        console.print(f"  [red]{t.exit_line()}[/red]\n")
