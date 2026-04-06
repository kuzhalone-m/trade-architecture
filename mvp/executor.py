# =============================================================================
# executor.py
#
# Takes an approved signal dict and places a real order on MT5.
# Does NOT generate signals. Does NOT make risk decisions.
# Just executes — cleanly, with anti-detection, and logs everything.
#
# Usage:
#   from executor import Executor
#   executor = Executor(risk_engine)
#   result = executor.execute(signal)
# =============================================================================

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

import config
from risk_engine import RiskEngine

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [EXECUTOR]  %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_PATH}executor.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("Executor")


# =============================================================================
class Executor:

    def __init__(self, risk_engine: RiskEngine):
        self.engine       = risk_engine
        self.trade_log    = f"{config.LOG_PATH}trades.json"
        self._ensure_trade_log()

    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------

    def execute(self, signal: dict) -> dict:
        """
        Full execution pipeline — 8 steps.
        Returns a result dict with status and fill details.

        Signal dict required fields:
            symbol, direction, entry_price, stop_loss,
            take_profit, confidence, signal_id
        """
        sid = signal.get("signal_id", "unknown")
        log.info(f"[{sid}] Received signal — {signal['direction']} {signal['symbol']} @ {signal['entry_price']}")

        # ------------------------------------------------------------------
        # STEP 1 — Pre-trade risk check
        # ------------------------------------------------------------------
        approved, reason = self.engine.check_signal(signal)
        if not approved:
            log.warning(f"[{sid}] Blocked by risk engine — {reason}")
            return self._result("blocked", signal, reason=reason)

        # ------------------------------------------------------------------
        # STEP 2 — Calculate lot size
        # ------------------------------------------------------------------
        lot = self._calculate_lot(signal)
        if lot is None:
            log.error(f"[{sid}] Lot size calculation failed — skipping")
            return self._result("error", signal, reason="lot calculation failed")

        # ------------------------------------------------------------------
        # STEP 3 — Apply lot jitter (anti-detection)
        # ------------------------------------------------------------------
        jitter  = random.uniform(-config.LOT_JITTER, config.LOT_JITTER)
        lot     = round(lot * (1 + jitter), 2)
        lot     = max(lot, 0.01)   # MT5 minimum lot
        log.info(f"[{sid}] Lot size: {lot} (jitter: {jitter:+.1%})")

        # Halve lot size if warning is active
        if self.engine.warning_active():
            lot = max(round(lot * 0.5, 2), 0.01)
            log.warning(f"[{sid}] WARNING active — lot halved to {lot}")

        # ------------------------------------------------------------------
        # STEP 4 — Apply entry delay (anti-detection)
        # ------------------------------------------------------------------
        delay = random.uniform(config.ENTRY_DELAY_MIN, config.ENTRY_DELAY_MAX)
        log.info(f"[{sid}] Waiting {delay:.1f}s before placing order...")
        time.sleep(delay)

        # ------------------------------------------------------------------
        # STEP 5 — Place limit order
        # ------------------------------------------------------------------
        order_result = self._place_order(signal, lot)
        if order_result is None:
            log.error(f"[{sid}] Order placement failed")
            return self._result("error", signal, lot=lot, reason="order_send failed")

        ticket = order_result.order
        log.info(f"[{sid}] Order placed — ticket: {ticket}")

        # ------------------------------------------------------------------
        # STEP 6 — Log the order immediately
        # ------------------------------------------------------------------
        self._log_trade({
            "signal_id":   sid,
            "ticket":      ticket,
            "symbol":      signal["symbol"],
            "direction":   signal["direction"],
            "lot":         lot,
            "entry_price": signal["entry_price"],
            "stop_loss":   signal["stop_loss"],
            "take_profit": signal["take_profit"],
            "confidence":  signal["confidence"],
            "status":      "pending",
            "placed_at":   datetime.now(timezone.utc).isoformat()
        })

        # ------------------------------------------------------------------
        # STEP 7 — Monitor for fill (poll for up to 60 seconds)
        # ------------------------------------------------------------------
        fill = self._wait_for_fill(ticket, sid)

        # ------------------------------------------------------------------
        # STEP 8 — Report fill back to risk engine and log
        # ------------------------------------------------------------------
        if fill:
            slippage = abs(fill["fill_price"] - signal["entry_price"])
            log.info(
                f"[{sid}] FILLED — price: {fill['fill_price']} | "
                f"slippage: {slippage:.2f} | lot: {lot}"
            )
            self._update_trade_log(ticket, {
                "status":     "filled",
                "fill_price": fill["fill_price"],
                "fill_time":  fill["fill_time"],
                "slippage":   slippage
            })
            return self._result("filled", signal, lot=lot, ticket=ticket, fill=fill)
        else:
            # Not filled in 60 seconds — cancel the order
            self._cancel_order(ticket, sid)
            self._update_trade_log(ticket, {"status": "cancelled"})
            return self._result("cancelled", signal, lot=lot, ticket=ticket,
                                reason="not filled within 60 seconds")

    # -------------------------------------------------------------------------
    # LOT SIZE CALCULATION
    # -------------------------------------------------------------------------

    def _calculate_lot(self, signal: dict):
        """
        lot = (equity * RISK_PER_TRADE) / (SL distance in price * contract size)

        For XAUUSD on most brokers:
          1 lot = 100 oz
          pip value = $1 per 0.01 price move per 0.01 lot
          contract size = 100

        Formula: lot = risk_amount / (sl_pips * pip_value_per_lot)
        where pip_value_per_lot for XAUUSD = 1.0 per pip per lot
        """
        try:
            equity     = self.engine.current_equity
            risk_amt   = equity * config.RISK_PER_TRADE
            sl_dist    = abs(signal["entry_price"] - signal["stop_loss"])

            if sl_dist == 0:
                log.error("SL distance is zero — cannot calculate lot size")
                return None

            # Get contract size from MT5 symbol info
            sym_info = mt5.symbol_info(signal["symbol"])
            if sym_info is None:
                log.error(f"Symbol info not available for {signal['symbol']}")
                return None

            # tick_value = value of 1 tick move for 1 lot
            # tick_size  = size of 1 tick
            tick_value = sym_info.trade_tick_value
            tick_size  = sym_info.trade_tick_size

            if tick_size == 0:
                log.error("tick_size is zero")
                return None

            pip_value_per_lot = (tick_value / tick_size) * tick_size
            lot = risk_amt / (sl_dist * (tick_value / tick_size))

            # Clamp to broker limits
            lot = max(lot, sym_info.volume_min)
            lot = min(lot, sym_info.volume_max)

            # Round to broker's lot step
            step = sym_info.volume_step
            lot  = round(round(lot / step) * step, 2)

            log.info(
                f"Lot calc — equity: {equity:.2f} | risk: {risk_amt:.2f} | "
                f"SL dist: {sl_dist:.2f} | lot: {lot}"
            )
            return lot

        except Exception as e:
            log.error(f"Lot calculation error: {e}")
            return None

    # -------------------------------------------------------------------------
    # PLACE ORDER
    # -------------------------------------------------------------------------

    def _place_order(self, signal: dict, lot: float):
        """
        Places a limit order with a small entry offset (anti-detection).
        BUY  LIMIT: entry slightly below signal level
        SELL LIMIT: entry slightly above signal level
        """
        direction  = signal["direction"].upper()
        symbol     = signal["symbol"]
        entry      = signal["entry_price"]
        sl         = signal["stop_loss"]
        tp         = signal["take_profit"]

        # Anti-detection: offset entry by 0.5-1.5 pips from signal level
        tick    = mt5.symbol_info_tick(symbol)
        point   = mt5.symbol_info(symbol).point
        offset  = random.uniform(0.5, 1.5) * point * 10  # ~0.5-1.5 pips

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
            entry      = round(entry - offset, mt5.symbol_info(symbol).digits)
        else:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT
            entry      = round(entry + offset, mt5.symbol_info(symbol).digits)

        request = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        entry,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        20260101,   # our system magic number
            "comment":      f"mvp_{signal['signal_id'][:10]}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        result = mt5.order_send(request)

        if result is None:
            log.error(f"order_send returned None — {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"Order failed — retcode: {result.retcode} | comment: {result.comment}")
            return None

        return result

    # -------------------------------------------------------------------------
    # WAIT FOR FILL
    # -------------------------------------------------------------------------

    def _wait_for_fill(self, ticket: int, sid: str) -> dict | None:
        """
        Poll order status every 2 seconds for up to 60 seconds.
        Returns fill details if filled, None if not filled.
        """
        log.info(f"[{sid}] Waiting for fill on ticket {ticket}...")
        deadline = time.time() + 60

        while time.time() < deadline:
            # Check if it became an open position (i.e. filled)
            positions = mt5.positions_get(ticket=ticket)
            if positions:
                pos = positions[0]
                return {
                    "fill_price": pos.price_open,
                    "fill_time":  datetime.now(timezone.utc).isoformat()
                }

            # Check if order is still pending
            orders = mt5.orders_get(ticket=ticket)
            if not orders:
                # Order disappeared without becoming a position — failed
                log.warning(f"[{sid}] Order {ticket} disappeared — may have been rejected")
                return None

            time.sleep(2)

        log.warning(f"[{sid}] Order {ticket} not filled within 60 seconds")
        return None

    # -------------------------------------------------------------------------
    # CANCEL ORDER
    # -------------------------------------------------------------------------

    def _cancel_order(self, ticket: int, sid: str):
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order":  ticket,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"[{sid}] Order {ticket} cancelled.")
        else:
            retcode = result.retcode if result else "no result"
            log.error(f"[{sid}] Failed to cancel order {ticket} — retcode: {retcode}")

    # -------------------------------------------------------------------------
    # TRADE LOG
    # -------------------------------------------------------------------------

    def _ensure_trade_log(self):
        if not os.path.exists(self.trade_log):
            with open(self.trade_log, "w") as f:
                json.dump([], f)

    def _log_trade(self, entry: dict):
        try:
            with open(self.trade_log) as f:
                trades = json.load(f)
            trades.append(entry)
            with open(self.trade_log, "w") as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            log.error(f"Failed to log trade: {e}")

    def _update_trade_log(self, ticket: int, updates: dict):
        try:
            with open(self.trade_log) as f:
                trades = json.load(f)
            for trade in trades:
                if trade.get("ticket") == ticket:
                    trade.update(updates)
                    break
            with open(self.trade_log, "w") as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            log.error(f"Failed to update trade log: {e}")

    # -------------------------------------------------------------------------
    # RESULT HELPER
    # -------------------------------------------------------------------------

    def _result(self, status: str, signal: dict, lot=None, ticket=None, fill=None, reason=None) -> dict:
        return {
            "status":     status,
            "signal_id":  signal.get("signal_id"),
            "symbol":     signal.get("symbol"),
            "direction":  signal.get("direction"),
            "lot":        lot,
            "ticket":     ticket,
            "fill":       fill,
            "reason":     reason,
            "timestamp":  datetime.now(timezone.utc).isoformat()
        }
