# =============================================================================
# config.py — All settings live here. Nothing is hardcoded anywhere else.
# =============================================================================

# --- MT5 Connection ----------------------------------------------------------
MT5_LOGIN    = 51522
MT5_PASSWORD = "Test123@"
MT5_SERVER   = "AurumMarkets-Demo"

# --- Instrument --------------------------------------------------------------
SYMBOL       = "XAUUSD"

# --- Risk Settings -----------------------------------------------------------
RISK_PER_TRADE    = 0.005   # 0.5% of equity per trade
MAX_POSITIONS     = 2       # Max concurrent open positions

# Drawdown thresholds (as decimals — 0.035 = 3.5%)
DAILY_DD_WARNING  = 0.020   # 2.0%  → halve position sizes, alert
DAILY_DD_HALT     = 0.035   # 3.5%  → close all, stop trading
MAX_DD_HALT       = 0.070   # 7.0%  → close all, permanent halt

# --- Anti-Detection ----------------------------------------------------------
ENTRY_DELAY_MIN   = 0.8     # Minimum seconds before placing order
ENTRY_DELAY_MAX   = 2.5     # Maximum seconds before placing order
LOT_JITTER        = 0.10    # +/- 10% randomisation on lot size

# --- Session Windows (UTC) ---------------------------------------------------
ASIAN_SESSION_END_UTC  = "07:00"   # Asian range calculated up to this time
LONDON_SESSION_END_UTC = "10:00"   # No new signals after this time

# --- Prop Firm Reset Time (UTC) ----------------------------------------------
# Pipstone = 17:00 EST = 22:00 UTC
DAILY_RESET_UTC   = "22:00"

# --- Paths -------------------------------------------------------------------
LOG_PATH          = "./logs/"
STATE_FILE        = "./logs/state.json"
HALT_FILE         = "./logs/HALTED"
WARNING_FILE      = "./logs/WARNING"
NEWS_FILE         = "./news_events.json"

# --- News Blackout -----------------------------------------------------------
NEWS_BLACKOUT_MINS = 15

# --- Watchdog ----------------------------------------------------------------
WATCHDOG_INTERVAL  = 1      # Seconds between each equity check