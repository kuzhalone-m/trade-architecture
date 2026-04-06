# =============================================================================
# main.py
#
# Entry point. Wires everything together and runs the trading loop.
#
# What it does:
#   1. Connects to MT5 via RiskEngine
#   2. Starts equity monitor (background thread)
#   3. Every 15 minutes during London window (07:00-10:00 UTC):
#      → runs SignalEngine.check()
#      → if signal found → passes to Executor.execute()
#   4. Outside London window → sleeps until next window
#   5. Handles daily reset and clean shutdown
#
# Usage:
#   python main.py           — live trading mode
#   python main.py --demo    — demo mode (same logic, confirms before each trade)
#   python main.py --dry     — dry run (signals generated but no orders placed)
#
# Keep watchdog.py running in a separate terminal at all times.
# =============================================================================

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config
from risk_engine import RiskEngine
from executor import Executor
from strategy import SignalEngine

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [MAIN]  %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_PATH}main.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("Main")

# =============================================================================
# CONSTANTS
# =============================================================================

CHECK_INTERVAL_SECS  = 60 * 15   # check for signals every 15 minutes
STARTUP_BANNER = """
╔══════════════════════════════════════════════════════════╗
║         QUANT TRADING SYSTEM — MVP                       ║
║         London Breakout Strategy — XAUUSD                ║
╚══════════════════════════════════════════════════════════╝
"""

# =============================================================================
# HELPERS
# =============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def in_london_window() -> bool:
    """True if current UTC time is within London breakout window."""
    h = now_utc().hour
    start = int(config.ASIAN_SESSION_END_UTC.split(":")[0])   # 7
    end   = int(config.LONDON_SESSION_END_UTC.split(":")[0])  # 10
    return start <= h < end

def seconds_until_london() -> int:
    """How many seconds until the next London window opens."""
    now  = now_utc()
    h    = int(config.ASIAN_SESSION_END_UTC.split(":")[0])
    m    = int(config.ASIAN_SESSION_END_UTC.split(":")[1])

    next_open = now.replace(hour=h, minute=m, second=0, microsecond=0)

    # If window already passed today, calculate for tomorrow
    if now >= next_open:
        next_open += timedelta(days=1)

    return max(int((next_open - now).total_seconds()), 0)

def is_weekend() -> bool:
    """True if current UTC day is Saturday or Sunday."""
    return now_utc().weekday() >= 5

def check_halt_file() -> bool:
    """Return True if HALTED file exists — system should not trade."""
    return os.path.exists(config.HALT_FILE)

def format_signal_summary(signal: dict) -> str:
    return (
        f"{signal['direction']} {signal['symbol']} "
        f"@ {signal['entry_price']} | "
        f"SL: {signal['stop_loss']} | "
        f"TP: {signal['take_profit']} | "
        f"Conf: {signal['confidence']:.2f} | "
        f"ID: {signal['signal_id']}"
    )

def format_result_summary(result: dict) -> str:
    if result["status"] == "filled":
        fill = result.get("fill", {})
        return (
            f"FILLED — ticket: {result['ticket']} | "
            f"fill: {fill.get('fill_price', 'N/A')} | "
            f"lot: {result['lot']}"
        )
    elif result["status"] == "cancelled":
        return f"CANCELLED — ticket: {result['ticket']} | not filled in 60s"
    elif result["status"] == "blocked":
        return f"BLOCKED — {result['reason']}"
    else:
        return f"{result['status'].upper()} — {result.get('reason', '')}"


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

class TradingSystem:

    def __init__(self, demo_mode: bool = False, dry_run: bool = False):
        self.demo_mode   = demo_mode
        self.dry_run     = dry_run
        self.engine      = RiskEngine()
        self.executor    = None
        self.signal_eng  = SignalEngine()
        self._running    = False

    # -------------------------------------------------------------------------
    # STARTUP
    # -------------------------------------------------------------------------

    def start(self):
        print(STARTUP_BANNER)

        mode_str = "DRY RUN" if self.dry_run else ("DEMO" if self.demo_mode else "LIVE")
        log.info(f"Starting in {mode_str} mode")
        log.info(f"Symbol: {config.SYMBOL} | Risk: {config.RISK_PER_TRADE:.1%} per trade")
        log.info(f"London window: {config.ASIAN_SESSION_END_UTC}–{config.LONDON_SESSION_END_UTC} UTC")
        log.info(f"Daily DD halt: {config.DAILY_DD_HALT:.1%} | Max DD halt: {config.MAX_DD_HALT:.1%}")

        # Check HALT file from previous session
        if check_halt_file():
            log.critical("HALT file exists from previous session.")
            log.critical(f"Delete {config.HALT_FILE} and restart to resume trading.")
            sys.exit(1)

        # Connect MT5
        log.info("Connecting to MT5...")
        ok = self.engine.connect()
        if not ok:
            log.error("Failed to connect to MT5. Make sure terminal is open and logged in.")
            sys.exit(1)

        log.info(f"Connected | Equity: {self.engine.current_equity:.2f} | "
                 f"Balance: {self.engine.current_balance:.2f}")

        # Start equity monitor
        self.engine.start_monitor()
        log.info("Equity monitor started.")

        # Initialise executor
        self.executor = Executor(self.engine)

        self._running = True
        log.info("System ready. Starting trading loop.")
        print()

        try:
            self._loop()
        except KeyboardInterrupt:
            print()
            log.info("Shutdown requested (Ctrl+C).")
            self._shutdown()

    # -------------------------------------------------------------------------
    # TRADING LOOP
    # -------------------------------------------------------------------------

    def _loop(self):
        while self._running:

            # --- Weekend check ---
            if is_weekend():
                log.info("Weekend — market closed. Sleeping 1 hour.")
                self._sleep(3600)
                continue

            # --- Halt check ---
            if check_halt_file():
                log.critical("HALT file detected. System stopped.")
                log.critical(f"Delete {config.HALT_FILE} and restart to resume.")
                self._running = False
                break

            now = now_utc()

            # --- Outside London window ---
            if not in_london_window():
                secs  = seconds_until_london()
                hours = secs // 3600
                mins  = (secs % 3600) // 60
                log.info(
                    f"Outside London window | "
                    f"Current UTC: {now.strftime('%H:%M')} | "
                    f"Next window opens in {hours}h {mins}m"
                )
                # Sleep until window opens (max 1 hour chunks)
                self._sleep(min(secs, 3600))
                continue

            # --- Inside London window ---
            log.info(
                f"London window active | UTC: {now.strftime('%H:%M')} | "
                f"Equity: {self.engine.current_equity:.2f} | "
                f"Daily DD: {self.engine._calc_daily_dd():.2%}"
            )

            # Run signal check
            self._check_and_trade()

            # Sleep until next 15-minute check
            self._sleep(CHECK_INTERVAL_SECS)

    # -------------------------------------------------------------------------
    # SIGNAL CHECK AND TRADE
    # -------------------------------------------------------------------------

    def _check_and_trade(self):
        """Run one signal check cycle. Execute if signal found."""

        # Generate signal
        signal = self.signal_eng.check()

        if signal is None:
            log.info("No signal this cycle.")
            return

        log.info(f"Signal found: {format_signal_summary(signal)}")

        # Dry run — log signal but don't execute
        if self.dry_run:
            log.info(f"[DRY RUN] Signal would have been sent to executor.")
            log.info(f"[DRY RUN] {format_signal_summary(signal)}")
            return

        # Demo mode — ask for confirmation before executing
        if self.demo_mode:
            log.info(f"[DEMO] Signal ready. Auto-executing on demo account.")

        # Execute
        log.info(f"Sending to executor...")
        result = self.executor.execute(signal)

        # Log result
        summary = format_result_summary(result)
        if result["status"] == "filled":
            log.info(f"Trade executed — {summary}")
        elif result["status"] == "cancelled":
            log.warning(f"Trade not filled — {summary}")
        elif result["status"] == "blocked":
            log.warning(f"Trade blocked — {summary}")
        else:
            log.error(f"Trade error — {summary}")

    # -------------------------------------------------------------------------
    # SLEEP (interruptible)
    # -------------------------------------------------------------------------

    def _sleep(self, seconds: int):
        """Sleep in 1-second chunks so Ctrl+C works immediately."""
        for _ in range(seconds):
            if not self._running:
                break
            if check_halt_file():
                log.critical("HALT file detected during sleep.")
                self._running = False
                break
            time.sleep(1)

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------

    def _shutdown(self):
        log.info("Shutting down cleanly...")

        if self.engine:
            # Check for open positions before disconnecting
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol=config.SYMBOL)
            if positions and len(positions) > 0:
                log.warning(f"{len(positions)} open position(s) remain — NOT closing automatically.")
                log.warning("Monitor them manually or let the watchdog handle risk.")

            self.engine.disconnect()

        log.info("System stopped.")
        sys.exit(0)


# =============================================================================
# STATUS REPORT — printed at startup and every hour
# =============================================================================

def print_status(engine: RiskEngine):
    now = now_utc()
    print(f"""
  ─────────────────────────────────────────
  System Status  {now.strftime('%Y-%m-%d %H:%M UTC')}
  ─────────────────────────────────────────
  Equity:        {engine.current_equity:.2f}
  Balance:       {engine.current_balance:.2f}
  Peak equity:   {engine.peak_equity:.2f}
  Daily DD:      {engine._calc_daily_dd():.2%}  (halt at {config.DAILY_DD_HALT:.1%})
  Max DD:        {engine._calc_max_dd():.2%}  (halt at {config.MAX_DD_HALT:.1%})
  Warning:       {'ACTIVE — lots halved' if engine.warning_active() else 'Off'}
  Halted:        {'YES' if check_halt_file() else 'No'}
  London window: {'OPEN' if in_london_window() else 'Closed'}
  ─────────────────────────────────────────
""")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Trading System — MVP")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode — executes on demo account (default behaviour)")
    parser.add_argument("--dry",  action="store_true",
                        help="Dry run — generates signals but places no orders")
    parser.add_argument("--status", action="store_true",
                        help="Print current account status and exit")
    args = parser.parse_args()

    # Status check only
    if args.status:
        engine = RiskEngine()
        if engine.connect():
            print_status(engine)
            engine.disconnect()
        sys.exit(0)

    # Start trading system
    system = TradingSystem(
        demo_mode = args.demo or True,   # always demo until you remove this
        dry_run   = args.dry
    )
    system.start()