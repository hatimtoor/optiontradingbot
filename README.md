# Options Trading Signal Bot

A live signal generator and backtesting system for **standard options trading** (CALL / PUT) on US stocks and ETFs. Uses technical analysis, an ML model, and multiple accuracy filters to generate high-confidence signals backed by an 84.9% historical win rate across 93 backtested trades.

---

## Features

- **Live signals** — BUY CALL / BUY PUT / HOLD for any ticker with real-time data
- **Specific contract recommendations** — exact expiration, strike, bid/ask/mid price to enter at
- **ML filter** — GradientBoostingClassifier trained on historical trades (84.9% win rate)
- **4 accuracy filters:**
  - IV cap (skip when implied volatility > 35% — premium too expensive)
  - ADX filter (skip flat markets — no trend = no edge)
  - Market regime (SPY-based bull/bear/neutral — trade with the market)
  - Earnings filter (skip within 5 days of earnings — IV crush risk)
- **Sector ETF alignment** — confirms trade direction matches sector trend
- **ATR-based exits** — dynamic take profit (2× ATR) and stop loss (1× ATR)
- **Live scanner** — monitors 45+ tickers every N seconds, alerts on signals
- **Backtester** — full historical backtest with checkpoint/resume support
- **168 tickers** across tech, finance, healthcare, energy, consumer, ETFs

---

## Installation

```bash
cd "c:\option trading bot"
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, yfinance, pandas, numpy, rich, ta, scipy, pyarrow, scikit-learn

---

## Quick Start

```bash
# Get a live signal for any ticker right now
python main.py AAPL
python main.py TSLA
python main.py SPY

# Auto-refresh every 60 seconds (watch mode)
python main.py AAPL --watch 60

# Show more expiration dates
python main.py NVDA --expirations 5
```

---

## Live Scanner

Scans the full watchlist continuously, alerts only when a BUY CALL or BUY PUT fires.

```bash
# Scan 45 tickers every 90 seconds
python scanner.py --interval 90

# Watch specific tickers every 30 seconds
python scanner.py --tickers TSLA NVDA AMD COIN --interval 30

# Single scan then exit
python scanner.py --once

# Show errors and timeouts
python scanner.py --verbose

# Scan all 168 tickers (slow)
python scanner.py --all
```

---

## Backtesting

### Run the full backtest

```bash
# Download data + run backtest (first time)
python run_backtest.py

# Use cached data (faster)
python run_backtest.py --no-download

# Single ticker
python run_backtest.py --ticker AAPL --no-download

# Show full trade log
python run_backtest.py --no-download --log

# Show best and worst trades
python run_backtest.py --no-download --best-worst

# Clear checkpoints and start over
python run_backtest.py --reset
```

### Train the ML model (do this after first backtest)

```bash
python ml_signal.py --train
```

Then re-run the backtest — the ML filter will now be active.

---

## Backtest Results (with all filters active)

| Metric            | Value         |
|-------------------|---------------|
| Total trades      | 93            |
| Win rate          | **84.9%**     |
| CALL win rate     | 83.1% (59/71) |
| PUT win rate      | 90.9% (20/22) |
| Avg return/trade  | +112.5%       |
| Avg P&L/trade     | +$1,284       |
| Total P&L         | +$119,446     |

**Exit breakdown:** 75.3% take profit (ATR), 14.0% stop loss (ATR), 9.7% time stop, 1.1% hard stop

---

## Strategy

Each signal is scored from **-100 to +100** using:

| Indicator       | Weight  | Logic                                      |
|-----------------|---------|--------------------------------------------|
| RSI(14)         | ±25 pts | < 30 oversold (bullish), > 70 overbought   |
| MACD(12,26,9)   | ±20 pts | Histogram crossover = stronger signal      |
| Bollinger %B    | ±20 pts | Near lower band = oversold bounce          |
| EMA(20/50)      | ±15 pts | Price above EMA20 > EMA50 = uptrend        |
| Volume ratio    | amplify | > 1.5× avg = 15% score boost              |

**Signal fires when:** score ≥ +30 (BUY CALL) or ≤ -30 (BUY PUT)

**Additional filters (all must pass):**
1. IV ≤ 35% — option premium affordable
2. ADX ≥ 15 — trend is strong enough
3. Market regime — no trades against SPY trend
4. No earnings within 5 days
5. Sector ETF aligned with trade direction
6. ML model predicts ≥ 52% win probability

**Exit rules:**
- Take profit: stock moves 2× ATR in trade direction
- Stop loss: stock moves 1× ATR against trade direction
- Hard stop: option loses 80% of premium
- Time stop: 5 days before expiration

---

## File Structure

```
option trading bot/
├── main.py               # CLI — live signal for a single ticker
├── scanner.py            # Live multi-ticker scanner
├── run_backtest.py       # Full backtest pipeline
├── backtester.py         # Core backtesting engine
├── backtest_report.py    # Results display
├── ml_signal.py          # ML model training and inference
├── signal_engine.py      # Scoring and signal classification
├── signal_filters.py     # All accuracy filters
├── technical_analysis.py # Indicator computation (live)
├── data_fetcher.py       # Live yfinance data fetcher
├── data_downloader.py    # Bulk historical data downloader
├── options_pricer.py     # Black-Scholes option pricing
├── display.py            # Rich terminal output
├── tickers.py            # 168-ticker watchlist
└── requirements.txt
```

---

## Disclaimer

This tool is for **informational and educational purposes only**. It is NOT financial advice. Options trading involves significant risk of loss, including the potential loss of the entire premium paid. Past backtest performance does not guarantee future results. Always do your own research and consult a licensed financial advisor before trading.
