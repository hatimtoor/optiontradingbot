"""
Rich terminal output for the options signal bot.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box
console = Console()


# ── Color helpers ──────────────────────────────────────────────────────────────

DIRECTION_STYLE = {
    "CALL": "bold green",
    "PUT": "bold red",
    "HOLD": "bold yellow",
}

CONFIDENCE_STYLE = {
    "HIGH": "bold bright_green",
    "MEDIUM": "bold yellow",
    "LOW": "dim white",
}

DIRECTION_EMOJI = {
    "CALL": "^",
    "PUT": "v",
    "HOLD": "-",
}


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}%"


def _fmt_price(val) -> str:
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


# ── Ticker info banner ─────────────────────────────────────────────────────────

def print_ticker_header(ticker: str, info: dict) -> None:
    name = info.get("name", ticker.upper())
    price = info.get("current_price")
    high52 = info.get("52w_high")
    low52 = info.get("52w_low")
    beta = info.get("beta")
    sector = info.get("sector", "N/A")

    lines = [
        f"[bold cyan]{name}[/bold cyan]  ([dim]{ticker.upper()}[/dim])",
        f"Price: [bold white]{_fmt_price(price)}[/bold white]   "
        f"52W: [dim]{_fmt_price(low52)} to {_fmt_price(high52)}[/dim]",
        f"Sector: [dim]{sector}[/dim]   Beta: [dim]{beta if beta else 'N/A'}[/dim]",
    ]
    console.print(Panel("\n".join(lines), title="[bold]Ticker Info[/bold]", border_style="cyan"))


# ── Technical indicators table ─────────────────────────────────────────────────

def print_ta_table(ta: dict) -> None:
    table = Table(title="Technical Indicators", box=box.SIMPLE_HEAVY, border_style="blue")
    table.add_column("Indicator", style="bold white", width=18)
    table.add_column("Value", justify="right", width=14)
    table.add_column("Signal", width=20)

    rsi = ta["rsi"]
    rsi_sig = (
        "[green]Oversold[/green]" if rsi < 30 else
        "[red]Overbought[/red]" if rsi > 70 else
        "[dim]Neutral[/dim]"
    )
    table.add_row("RSI (14)", f"{rsi:.2f}", rsi_sig)

    macd = ta["macd"]
    macd_sig_val = ta["macd_signal"]
    macd_hist = macd - macd_sig_val
    macd_sig = "[green]Bullish[/green]" if macd_hist > 0 else "[red]Bearish[/red]"
    table.add_row("MACD", f"{macd:.4f}", macd_sig)
    table.add_row("MACD Signal", f"{macd_sig_val:.4f}", "")
    table.add_row("MACD Histogram", f"{macd_hist:.4f}", macd_sig)

    pct_b = ta["bb_pct_b"]
    bb_sig = (
        "[green]Near Lower Band[/green]" if pct_b < 0.1 else
        "[red]Near Upper Band[/red]" if pct_b > 0.9 else
        "[dim]Mid-range[/dim]"
    )
    table.add_row("BB %B", f"{pct_b:.3f}", bb_sig)

    table.add_row("EMA 20", _fmt_price(ta["ema_20"]), "")
    table.add_row("EMA 50", _fmt_price(ta["ema_50"]), "")
    table.add_row("ATR (14)", _fmt_price(ta["atr"]), "")

    vol = ta["vol_ratio"]
    vol_sig = (
        "[green]High Volume[/green]" if vol > 1.5 else
        "[dim]Low Volume[/dim]" if vol < 0.7 else
        "[dim]Normal[/dim]"
    )
    table.add_row("Volume Ratio", f"{vol:.2f}x", vol_sig)

    console.print(table)


# ── Main signal panel ──────────────────────────────────────────────────────────

def print_signal(signal: dict, ticker: str) -> None:
    direction = signal["direction"]
    confidence = signal["confidence"]
    score = signal["score"]

    style = DIRECTION_STYLE[direction]
    conf_style = CONFIDENCE_STYLE[confidence]
    icon = DIRECTION_EMOJI[direction]

    # Big signal header
    signal_text = Text(justify="center")
    signal_text.append(f"\n  {icon}  SIGNAL: ", style="bold white")
    signal_text.append(f"BUY {direction}  " if direction != "HOLD" else "  HOLD  ", style=style)
    signal_text.append(f"\n  Confidence: ", style="bold white")
    signal_text.append(f"{confidence}", style=conf_style)
    signal_text.append(f"   Score: {score:+d}/100\n", style="dim")

    console.print(Panel(signal_text, title=f"[bold]{ticker.upper()} Options Signal[/bold]",
                        border_style=style.split()[-1]))

    # Reasons
    console.print("\n[bold]Signal Rationale:[/bold]")
    for r in signal["reasons"]:
        bullet = "[green]+[/green]" if "bullish" in r.lower() or "oversold" in r.lower() else \
                 "[red]-[/red]" if "bearish" in r.lower() or "overbought" in r.lower() else \
                 "[yellow]~[/yellow]"
        console.print(f"  {bullet} {r}")


# ── Recommended contracts table ────────────────────────────────────────────────

def print_contracts(contracts: list[dict], direction: str) -> None:
    if not contracts:
        if direction == "HOLD":
            console.print("\n[yellow]No contracts shown - signal is HOLD (no directional trade recommended).[/yellow]")
        else:
            console.print("\n[yellow]No liquid contracts found matching criteria.[/yellow]")
        return

    style = DIRECTION_STYLE.get(direction, "white")
    table = Table(
        title=f"Recommended {direction} Contracts",
        box=box.ROUNDED,
        border_style=style.split()[-1],
        show_lines=True,
    )

    table.add_column("Expiration", style="cyan", width=13)
    table.add_column("Strike", justify="right", width=10)
    table.add_column("Type", justify="center", width=6)
    table.add_column("Bid", justify="right", width=8)
    table.add_column("Ask", justify="right", width=8)
    table.add_column("Mid", justify="right", style="bold", width=8)
    table.add_column("IV %", justify="right", width=7)
    table.add_column("Volume", justify="right", width=8)
    table.add_column("OI", justify="right", width=8)
    table.add_column("ITM", justify="center", width=5)

    for c in contracts:
        itm_mark = "[green]Y[/green]" if c["itm"] else "[dim]N[/dim]"
        iv_str = _fmt_pct(c["iv"]) if c["iv"] else "N/A"
        table.add_row(
            c["expiration"],
            _fmt_price(c["strike"]),
            f"[{style.split()[-1]}]{c['type']}[/{style.split()[-1]}]",
            _fmt_price(c["bid"]),
            _fmt_price(c["ask"]),
            _fmt_price(c["mid"]),
            iv_str,
            str(c["volume"]),
            str(c["open_interest"]),
            itm_mark,
        )

    console.print()
    console.print(table)


# ── Disclaimer ─────────────────────────────────────────────────────────────────

def print_disclaimer() -> None:
    console.print("[dim]" + "-" * 78 + "[/dim]")
    console.print(
        "[dim]DISCLAIMER: This tool provides signals for informational and educational "
        "purposes only. It is NOT financial advice. Options trading involves significant "
        "risk of loss. Always do your own research and consult a licensed financial "
        "advisor before trading.[/dim]"
    )
    console.print("[dim]" + "-" * 78 + "[/dim]")
