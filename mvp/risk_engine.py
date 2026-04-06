# =============================================================================
# risk_engine.py
#
# Direct MT5 connection — no MetaApi required.
# MT5 terminal must be open and logged in before running this.
#
# Three jobs:
#   1. connect()        — initialise MT5, verify symbol, load state
#   2. check_signal()   — pre-trade validation, returns (bool, reason)
#   3. start_monitor()  — background thread, checks equity every second
#   4. close_all()      — closes every open position at market
#
# Run directly to test connection:
#   python risk_engine.py
# =============================================================================

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5

import config

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_PATH}risk_engine.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("RiskEngine")


# =============================================================================
class RiskEngine:

    def __init__(self):
        self.peak_equity      = None
        self.day_start_equity = None
        self.current_equity   = None
        self.current_balance  = None

        self._halted          = False
        self._warning_active  = False
        self._monitor_thread  = None
        self._stop_monitor    = threading.Event()

        self._load_state()

    # -------------------------------------------------------------------------
    # CONNECTION
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Initialise MT5. If MT5_LOGIN is set in config, logs in automatically.
        Otherwise uses whatever account is already logged in on the terminal.
        Returns True on success, False on failure.
        """
        log.info("Initialising MT5...")

        if config.MT5_LOGIN:
            ok = mt5.initialize(
                login=config.MT5_LOGIN,
                password=config.MT5_PASSWORD,
                server=config.MT5_SERVER
            )
        else:
            ok = mt5.initialize()

        if not ok:
            log.error(f"MT5 initialise failed: {mt5.last_error()}")
            return False

        # Verify the symbol is available
        info = mt5.symbol_info(config.SYMBOL)
        if info is None:
            log.error(f"Symbol {config.SYMBOL} not found. Make sure it is visible in MT5 Market Watch.")
            mt5.shutdown()
            return False

        if not info.visible:
            mt5.symbol_select(config.SYMBOL, True)
            log.info(f"{config.SYMBOL} added to Market Watch.")

        # Get initial account state
        acct = mt5.account_info()
        if acct is None:
            log.error("Could not fetch account info.")
            mt5.shutdown()
            return False

        self.current_equity  = acct.equity
        self.current_balance = acct.balance

        # First run — initialise state from live values
        if self.peak_equity is None:
            self.peak_equity = self.current_equity
            log.info(f"First run — peak equity set to {self.peak_equity:.2f}")

        if self.day_start_equity is None:
            self.day_start_equity = self.current_equity
            log.info(f"First run — day start equity set to {self.day_start_equity:.2f}")

        self._save_state()

        log.info(f"MT5 connected | Account: {acct.login} | Broker: {acct.company}")
        log.info(f"Balance: {self.current_balance:.2f} | Equity: {self.current_equity:.2f}")
        return True

    def disconnect(self):
        self._stop_monitor.set()
        mt5.shutdown()
        log.info("MT5 disconnected.")

    # -------------------------------------------------------------------------
    # PRE-TRADE CHECK
    # -------------------------------------------------------------------------

    def check_signal(self, signal: dict) -> tuple[bool, str]:
        """
        Validate a signal before passing it to the Executor.
        Returns (True, "approved") or (False, "reason").

        Required signal fields:
            symbol, direction, entry_price, stop_loss,
            take_profit, confidence, signal_id
        """
        sid = signal.get("signal_id", "unknown")

        # 1. System halt check
        if self._halted or os.path.exists(config.HALT_FILE):
            return False, "SYSTEM HALTED — delete logs/HALTED file and restart to resume"

        # 2. Stop-loss present
        sl = signal.get("stop_loss", 0)
        if not sl or sl == 0:
            return False, "Rejected — no stop-loss in signal"

        # 3. Confidence threshold
        confidence = signal.get("confidence", 0)
        if confidence < 0.65:
            return False, f"Rejected — confidence {confidence:.2f} below 0.65"

        # 4. Refresh equity before checks
        self._refresh_equity()

        # 5. Daily drawdown check
        daily_dd = self._calc_daily_dd()
        if daily_dd >= config.DAILY_DD_HALT:
            self.close_all(f"Pre-trade: daily DD {daily_dd:.2%} hit halt level")
            return False, f"Rejected — daily DD {daily_dd:.2%} at or above halt level {config.DAILY_DD_HALT:.2%}"

        if daily_dd >= config.DAILY_DD_WARNING:
            self._set_warning(True)
            log.warning(f"[{sid}] Daily DD {daily_dd:.2%} — WARNING active")

        # 6. Max drawdown check
        max_dd = self._calc_max_dd()
        if max_dd >= config.MAX_DD_HALT:
            self.close_all(f"Pre-trade: max DD {max_dd:.2%} hit halt level")
            return False, f"Rejected — max DD {max_dd:.2%} at or above halt level {config.MAX_DD_HALT:.2%}"

        # 7. Position count check
        positions = mt5.positions_get(symbol=config.SYMBOL)
        count = len(positions) if positions else 0
        if count >= config.MAX_POSITIONS:
            return False, f"Rejected — {count} positions already open on {config.SYMBOL} (max {config.MAX_POSITIONS})"

        # 8. News blackout check
        blocked, event_name = self._check_news_blackout()
        if blocked:
            return False, f"Rejected — news blackout active ({event_name})"

        log.info(
            f"[{sid}] APPROVED | "
            f"Daily DD: {daily_dd:.2%} | Max DD: {max_dd:.2%} | "
            f"Open positions: {count} | Confidence: {confidence:.2f}"
        )
        return True, "approved"

    # -------------------------------------------------------------------------
    # EQUITY MONITOR (background thread)
    # -------------------------------------------------------------------------

    def start_monitor(self):
        """Start background thread — checks equity every second."""
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="EquityMonitor"
        )
        self._monitor_thread.start()
        log.info("Equity monitor started.")

    def _monitor_loop(self):
        last_log = 0
        while not self._stop_monitor.is_set():
            try:
                self._refresh_equity()
                self._check_daily_reset()
                self._evaluate_thresholds()

                # Log status every 60 seconds
                now = time.time()
                if now - last_log >= 60:
                    log.info(
                        f"Equity: {self.current_equity:.2f} | "
                        f"Daily DD: {self._calc_daily_dd():.2%} | "
                        f"Max DD: {self._calc_max_dd():.2%}"
                    )
                    last_log = now

            except Exception as e:
                log.error(f"Monitor loop error: {e}")

            time.sleep(config.WATCHDOG_INTERVAL)

    def _evaluate_thresholds(self):
        daily_dd = self._calc_daily_dd()
        max_dd   = self._calc_max_dd()

        # Warning
        if daily_dd >= config.DAILY_DD_WARNING and not self._warning_active:
            self._set_warning(True)
            log.warning(f"WARNING — daily DD {daily_dd:.2%}. Position sizes will be halved.")

        if daily_dd < config.DAILY_DD_WARNING and self._warning_active:
            self._set_warning(False)
            log.info("Daily DD recovered below warning level.")

        # Halt
        if daily_dd >= config.DAILY_DD_HALT:
            self.close_all(f"Monitor: daily DD {daily_dd:.2%} hit halt level")
            return

        if max_dd >= config.MAX_DD_HALT:
            self.close_all(f"Monitor: max DD {max_dd:.2%} hit halt level")

    # -------------------------------------------------------------------------
    # CLOSE ALL POSITIONS
    # -------------------------------------------------------------------------

    def close_all(self, reason: str):
        """
        Close every open position immediately at market.
        Safe to call multiple times — skips if already halted.
        """
        if self._halted:
            return

        self._halted = True
        self._write_halt_file(reason)
        log.critical(f"CLOSE ALL — reason: {reason}")

        positions = mt5.positions_get()
        if not positions:
            log.info("No open positions to close.")
            return

        for pos in positions:
            symbol    = pos.symbol
            volume    = pos.volume
            pos_type  = pos.type   # 0 = BUY, 1 = SELL

            # Close direction is opposite to open direction
            close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
            price      = mt5.symbol_info_tick(symbol).bid if pos_type == 0 else mt5.symbol_info_tick(symbol).ask

            request = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "position":  pos.ticket,
                "symbol":    symbol,
                "volume":    volume,
                "type":      close_type,
                "price":     price,
                "deviation": 20,
                "magic":     0,
                "comment":   "risk_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"Closed: {symbol} | {volume} lots | ticket {pos.ticket}")
            else:
                retcode = result.retcode if result else "no result"
                log.error(f"Failed to close ticket {pos.ticket} | retcode: {retcode}")

        log.critical(f"All positions closed. System HALTED.")

    # -------------------------------------------------------------------------
    # CALCULATIONS
    # -------------------------------------------------------------------------

    def _refresh_equity(self):
        acct = mt5.account_info()
        if acct is None:
            log.warning("Could not fetch account info — MT5 may have disconnected.")
            return

        self.current_equity  = acct.equity
        self.current_balance = acct.balance

        # Update high-water mark
        if self.current_equity > (self.peak_equity or 0):
            self.peak_equity = self.current_equity
            self._save_state()
            log.info(f"New peak equity: {self.peak_equity:.2f}")

    def _calc_daily_dd(self) -> float:
        if not self.day_start_equity or self.day_start_equity == 0:
            return 0.0
        return max((self.day_start_equity - self.current_equity) / self.day_start_equity, 0.0)

    def _calc_max_dd(self) -> float:
        if not self.peak_equity or self.peak_equity == 0:
            return 0.0
        return max((self.peak_equity - self.current_equity) / self.peak_equity, 0.0)

    def warning_active(self) -> bool:
        """Executor checks this to know whether to halve lot sizes."""
        return self._warning_active or os.path.exists(config.WARNING_FILE)

    # -------------------------------------------------------------------------
    # NEWS BLACKOUT
    # -------------------------------------------------------------------------

    def _check_news_blackout(self) -> tuple[bool, str]:
        """
        Reads news_events.json and blocks if within NEWS_BLACKOUT_MINS of any
        high-impact event.

        File format:
        [
          {"name": "US CPI",  "time_utc": "2026-03-21T13:30:00Z", "impact": "high"},
          {"name": "FOMC",    "time_utc": "2026-03-21T19:00:00Z", "impact": "high"}
        ]

        If the file does not exist, no blackout is applied.
        """
        if not os.path.exists(config.NEWS_FILE):
            return False, ""

        try:
            with open(config.NEWS_FILE) as f:
                events = json.load(f)
        except Exception:
            return False, ""

        now      = datetime.now(timezone.utc)
        blackout = timedelta(minutes=config.NEWS_BLACKOUT_MINS)

        for event in events:
            if event.get("impact", "").lower() != "high":
                continue
            try:
                event_time = datetime.fromisoformat(event["time_utc"].replace("Z", "+00:00"))
                diff       = event_time - now
                if timedelta(0) <= diff <= blackout:
                    mins_left = int(diff.total_seconds() / 60)
                    return True, f"{event['name']} in {mins_left} min"
            except Exception:
                continue

        return False, ""

    # -------------------------------------------------------------------------
    # DAILY RESET
    # -------------------------------------------------------------------------

    def _check_daily_reset(self):
        """Reset day_start_equity at the prop firm's daily reset time."""
        now      = datetime.now(timezone.utc)
        reset_h  = int(config.DAILY_RESET_UTC.split(":")[0])
        reset_m  = int(config.DAILY_RESET_UTC.split(":")[1])
        reset_dt = now.replace(hour=reset_h, minute=reset_m, second=0, microsecond=0)

        seconds_past = (now - reset_dt).total_seconds()
        if 0 <= seconds_past < config.WATCHDOG_INTERVAL:
            old = self.day_start_equity
            self.day_start_equity = self.current_equity
            self._set_warning(False)
            self._save_state()
            log.info(f"Daily reset — day start updated: {old:.2f} → {self.day_start_equity:.2f}")

    # -------------------------------------------------------------------------
    # STATE PERSISTENCE
    # -------------------------------------------------------------------------

    def _save_state(self):
        state = {
            "peak_equity":      self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "saved_at":         datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(config.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save state: {e}")

    def _load_state(self):
        if not os.path.exists(config.STATE_FILE):
            return
        try:
            with open(config.STATE_FILE) as f:
                state = json.load(f)
            self.peak_equity      = state.get("peak_equity")
            self.day_start_equity = state.get("day_start_equity")
            log.info(f"State loaded — peak: {self.peak_equity}, day_start: {self.day_start_equity}")
        except Exception as e:
            log.error(f"Failed to load state: {e}")

    # -------------------------------------------------------------------------
    # FLAG FILES
    # -------------------------------------------------------------------------

    def _write_halt_file(self, reason: str):
        try:
            with open(config.HALT_FILE, "w") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} — {reason}\n")
        except Exception as e:
            log.error(f"Failed to write HALT file: {e}")

    def _set_warning(self, active: bool):
        self._warning_active = active
        if active:
            try:
                Path(config.WARNING_FILE).touch()
            except Exception:
                pass
        else:
            try:
                if os.path.exists(config.WARNING_FILE):
                    os.remove(config.WARNING_FILE)
            except Exception:
                pass


# =============================================================================
# Run directly to test connection and pre-trade check
# =============================================================================
if __name__ == "__main__":
    engine = RiskEngine()

    ok = engine.connect()
    if not ok:
        print("\nFailed to connect. Make sure MT5 terminal is open and logged in.")
        exit(1)

    print(f"\n--- Account State ---")
    print(f"Equity:        {engine.current_equity:.2f}")
    print(f"Balance:       {engine.current_balance:.2f}")
    print(f"Peak equity:   {engine.peak_equity:.2f}")
    print(f"Day start:     {engine.day_start_equity:.2f}")
    print(f"Daily DD:      {engine._calc_daily_dd():.4%}")
    print(f"Max DD:        {engine._calc_max_dd():.4%}")

    print(f"\n--- Test Signal ---")
    test_signal = {
        "symbol":      "XAUUSD",
        "direction":   "BUY",
        "entry_price": 2000.00,
        "stop_loss":   1990.00,
        "take_profit": 2020.00,
        "confidence":  0.72,
        "signal_id":   "TEST_001"
    }
    approved, reason = engine.check_signal(test_signal)
    print(f"Result:  {'APPROVED' if approved else 'REJECTED'}")
    print(f"Reason:  {reason}")

    engine.disconnect()
