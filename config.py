# config.py — ETF universes and shared parameters

# Option A: Fixed Income / Commodities (existing)
OPTION_A_ETFS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

# Option B: Equity Sectors (new)
OPTION_B_ETFS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XLB", "XLRE", "XME", "IWF", "IWM", "XBI", "XSD", "XAR"
]

# Combined list for data fetching (all tickers)
ALL_TICKERS = OPTION_A_ETFS + OPTION_B_ETFS
