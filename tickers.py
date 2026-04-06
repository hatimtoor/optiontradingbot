"""
Curated list of ~200 highly liquid options tickers across all sectors.
These are chosen for high options volume, long price history, and sector diversity.
"""

TICKERS = [
    # ── Major Index ETFs (highest options volume in the market) ──
    "SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY", "SQQQ", "TQQQ",

    # ── Sector ETFs ──
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLB", "XLP", "XLU", "XLRE", "XLY",
    "GLD", "SLV", "GDX", "GDXJ", "USO", "TLT", "HYG", "LQD", "EEM", "FXI",

    # ── Mega-Cap Tech ──
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",

    # ── Large-Cap Tech / Semiconductors ──
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "TXN",
    "AVGO", "MRVL", "SMCI", "ARM", "TSM",

    # ── Software / Cloud ──
    "ORCL", "CRM", "ADBE", "CSCO", "IBM",
    "SNOW", "DDOG", "CRWD", "NET", "PANW", "ZS", "OKTA",
    "NOW", "WDAY", "VEEV",

    # ── Consumer Internet / Social ──
    "NFLX", "UBER", "LYFT", "SNAP", "PINS", "RDDT",
    "SHOP", "ETSY", "ABNB", "DASH", "RBLX",

    # ── Financials ──
    "JPM", "BAC", "C", "WFC", "GS", "MS", "BLK",
    "V", "MA", "AXP", "PYPL", "SQ", "COIN", "SOFI",
    "MET", "PRU", "AFL",

    # ── Healthcare / Pharma / Biotech ──
    "JNJ", "PFE", "MRNA", "BNTX", "ABBV", "UNH", "CVS",
    "LLY", "BMY", "MRK", "AMGN", "GILD", "BIIB", "REGN",
    "ISRG", "DHR", "BSX",

    # ── Energy ──
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "MPC", "VLO",

    # ── Consumer / Retail ──
    "WMT", "COST", "TGT", "HD", "LOW", "AMZN",
    "NKE", "LULU", "SBUX", "MCD", "YUM", "CMG",
    "DIS", "CMCSA", "PARA",

    # ── Telecom / Media ──
    "T", "VZ", "TMUS",

    # ── Industrials / Aerospace / Defense ──
    "BA", "GE", "CAT", "DE", "MMM", "HON",
    "LMT", "RTX", "NOC", "GD",

    # ── Autos / EV ──
    "F", "GM", "RIVN", "LCID",

    # ── Real Estate / REITs ──
    "AMT", "PLD", "SPG", "O",

    # ── Special / High-Volatility / Popular ──
    "BABA", "JD", "NIO", "XPEV", "LI",
    "GME", "AMC", "MSTR", "PLTR", "HOOD",
    "RKLB", "LUNR", "JOBY",

    # ── Commodities / Macro ──
    "IAU", "IBIT", "BITO",
]

# Remove duplicates while preserving order
seen = set()
TICKERS = [t for t in TICKERS if not (t in seen or seen.add(t))]
