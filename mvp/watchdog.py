# =============================================================================
# watchdog.py
#
# Runs as a SEPARATE process from risk_engine.py.
# Start it in a second terminal and leave it running.
#
# Job: check equity every second. If drawdown limits are breached,
# close all positions — even if the main process has crashed.
#
# Usage:
#   python watchdog.py
#
# Keep this running in a separate terminal whenever you are trading.
# =============================================================================

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5

import config

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [WATCHDOG]  %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_PATH}watchdog.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("Watchdog")


# =============================================================================
class Watchdog:

    def __init__(self):
        self.peak_equity      = None
        self.day_start_equity = None
        self.current_equity   = None

    # -------------------------------------------------------------------------
    # CONNECT
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
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
            log.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        acct = mt5.account_info()
        if acct is None:
            log.error("Could not fetch account info.")
            mt5.shutdown()
            return False

        self.current_equity = acct.equity
        self._load_state()

        # If state file exists, use saved peak and day_start.
        # If not, initialise from current equity.
        if self.peak_equity is None:
            self.peak_equity = self.current_equity
        if self.day_start_equity is None:
            self.day_start_equity = self.current_equity

        log.info(f"Connected | Account: {acct.login} | Equity: {self.current_equity:.2f}")
        log.info(f"Peak: {self.peak_equity:.2f} | Day start: {self.day_start_equity:.2f}")
        return True

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    def run(self):
        log.info("Watchdog running. Monitoring every second.")
        log.info(f"Halt levels — Daily DD: {config.DAILY_DD_HALT:.1%} | Max DD: {config.MAX_DD_HALT:.1%}")

        last_log = 0

        while True:
            try:
                self._refresh_equity()
                self._check_daily_reset()

                daily_dd = self._calc_daily_dd()
                max_dd   = self._calc_max_dd()

                # Log status every 60 seconds
                now = time.time()
                if now - last_log >= 60:
                    log.info(
                        f"Equity: {self.current_equity:.2f} | "
                        f"Daily DD: {daily_dd:.2%} | "
                        f"Max DD: {max_dd:.2%}"
                    )
                    last_log = now

                # Check halt conditions
                if daily_dd >= config.DAILY_DD_HALT:
                    self._close_all(f"Daily DD {daily_dd:.2%} hit halt level {config.DAILY_DD_HALT:.1%}")
                    break

                if max_dd >= config.MAX_DD_HALT:
                    self._close_all(f"Max DD {max_dd:.2%} hit halt level {config.MAX_DD_HALT:.1%}")
                    break

            except KeyboardInterrupt:
                log.info("Watchdog stopped by user.")
                break
            except Exception as e:
                log.error(f"Watchdog loop error: {e}")

            time.sleep(config.WATCHDOG_INTERVAL)

        log.info("Watchdog exiting.")
        mt5.shutdown()

    # -------------------------------------------------------------------------
    # CLOSE ALL
    # -------------------------------------------------------------------------

    def _close_all(self, reason: str):
        """
        Close every open position at market.
        Writes the HALT file so the main process also knows to stop.
        Safe to call even if the main process already closed positions.
        """
        log.critical(f"CLOSE ALL triggered — {reason}")

        # Write HALT file so risk_engine also stops accepting signals
        self._write_halt_file(reason)

        positions = mt5.positions_get()
        if not positions:
            log.info("No open positions found.")
            return

        for pos in positions:
            symbol   = pos.symbol
            volume   = pos.volume
            pos_type = pos.type  # 0 = BUY, 1 = SELL

            close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
            tick       = mt5.symbol_info_tick(symbol)
            price      = tick.bid if pos_type == 0 else tick.ask

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "position":     pos.ticket,
                "symbol":       symbol,
                "volume":       volume,
                "type":         close_type,
                "price":        price,
                "deviation":    20,
                "magic":        0,
                "comment":      "watchdog_close",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"Closed: {symbol} | {volume} lots | ticket {pos.ticket}")
            else:
                retcode = result.retcode if result else "no result"
                log.error(f"Failed to close ticket {pos.ticket} | retcode: {retcode}")

        log.critical(f"Done. System HALTED. Reason: {reason}")

    # -------------------------------------------------------------------------
    # CALCULATIONS
    # -------------------------------------------------------------------------

    def _refresh_equity(self):
        acct = mt5.account_info()
        if acct is None:
            log.warning("Could not fetch account info.")
            return

        self.current_equity = acct.equity

        if self.current_equity > (self.peak_equity or 0):
            self.peak_equity = self.current_equity
            self._save_state()

    def _calc_daily_dd(self) -> float:
        if not self.day_start_equity or self.day_start_equity == 0:
            return 0.0
        return max((self.day_start_equity - self.current_equity) / self.day_start_equity, 0.0)

    def _calc_max_dd(self) -> float:
        if not self.peak_equity or self.peak_equity == 0:
            return 0.0
        return max((self.peak_equity - self.current_equity) / self.peak_equity, 0.0)

    # -------------------------------------------------------------------------
    # DAILY RESET
    # -------------------------------------------------------------------------

    def _check_daily_reset(self):
        now     = datetime.now(timezone.utc)
        reset_h = int(config.DAILY_RESET_UTC.split(":")[0])
        reset_m = int(config.DAILY_RESET_UTC.split(":")[1])
        reset   = now.replace(hour=reset_h, minute=reset_m, second=0, microsecond=0)

        seconds_past = (now - reset).total_seconds()
        if 0 <= seconds_past < config.WATCHDOG_INTERVAL:
            old = self.day_start_equity
            self.day_start_equity = self.current_equity
            self._save_state()
            log.info(f"Daily reset — day start: {old:.2f} → {self.day_start_equity:.2f}")

    # -------------------------------------------------------------------------
    # STATE / FLAG FILES
    # -------------------------------------------------------------------------

    def _load_state(self):
        if not os.path.exists(config.STATE_FILE):
            return
        try:
            with open(config.STATE_FILE) as f:
                state = json.load(f)
            self.peak_equity      = state.get("peak_equity")
            self.day_start_equity = state.get("day_start_equity")
            log.info(f"State loaded — peak: {self.peak_equity} | day_start: {self.day_start_equity}")
        except Exception as e:
            log.error(f"Failed to load state: {e}")

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

    def _write_halt_file(self, reason: str):
        try:
            with open(config.HALT_FILE, "w") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} — {reason}\n")
        except Exception as e:
            log.error(f"Failed to write HALT file: {e}")


# =============================================================================
if __name__ == "__main__":
    watchdog = Watchdog()
    ok = watchdog.connect()
    if not ok:
        print("Failed to connect. Make sure MT5 is open and logged in.")
        exit(1)
    watchdog.run()
