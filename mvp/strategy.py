# =============================================================================
# signal.py
#
# XAUUSD London Breakout Strategy
#
# Rules:
#   1. Calculate Asian session range (00:00 - 07:00 UTC)
#   2. Watch for breakout of that range from 07:00 - 10:00 UTC
#   3. Confirm with EMA 55 direction on 15m bars
#   4. Optional: volume and 1h EMA 21 boost confidence
#   5. ATR-based SL and TP (1.5x ATR stop, 2.0x ATR target)
#   6. One trade per day maximum
#
# Usage:
#   from signal import SignalEngine
#   engine = SignalEngine()
#   signal = engine.check()   # returns signal dict or None
# =============================================================================

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import config

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [SIGNAL]  %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_PATH}signal.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("Strategy")


# =============================================================================
class SignalEngine:

    # Minimum Asian range in price points — skip if range too tight
    MIN_ASIAN_RANGE = 8.0

    # Signal threshold — only pass to executor if confidence >= this
    MIN_CONFIDENCE  = 0.65

    def __init__(self):
        self._traded_today  = False   # one trade per day flag
        self._last_trade_date = None  # date of last trade taken
        self._signal_count  = 0       # for signal ID generation

    # -------------------------------------------------------------------------
    # MAIN CHECK — call this every 15 minutes from 07:00 to 10:00 UTC
    # -------------------------------------------------------------------------

    def check(self) -> dict | None:
        """
        Run the full strategy logic.
        Returns a signal dict if all conditions are met, None otherwise.
        """
        now_utc = datetime.now(timezone.utc)

        # Reset daily trade flag at midnight UTC
        today = now_utc.date()
        if self._last_trade_date != today:
            self._traded_today    = False
            self._last_trade_date = today

        # Only run during London session window
        if not self._in_london_window(now_utc):
            log.info(f"Outside London window ({config.ASIAN_SESSION_END_UTC}–{config.LONDON_SESSION_END_UTC} UTC) — no check")
            return None

        # One trade per day
        if self._traded_today:
            log.info("Already traded today — skipping")
            return None

        # Get 15m bars — need enough for EMA 55 + ATR 14 + session range
        bars_15m = self._get_bars(mt5.TIMEFRAME_M15, count=200)
        if bars_15m is None or len(bars_15m) < 60:
            log.error("Not enough 15m bars")
            return None

        # --- Step 1: Asian range -------------------------------------------
        asian_high, asian_low, asian_range = self._calc_asian_range(bars_15m, now_utc)

        if asian_high is None:
            log.warning("Could not calculate Asian range")
            return None

        if asian_range < self.MIN_ASIAN_RANGE:
            log.info(f"Asian range {asian_range:.2f} < minimum {self.MIN_ASIAN_RANGE} — skip")
            return None

        log.info(f"Asian range — High: {asian_high:.2f} | Low: {asian_low:.2f} | Range: {asian_range:.2f}")

        # --- Step 2: Breakout check ----------------------------------------
        current_bar   = bars_15m.iloc[-1]
        previous_bar  = bars_15m.iloc[-2]
        close         = current_bar["close"]
        direction     = None

        # Only count a breakout if the CURRENT bar closes beyond the range
        # and the PREVIOUS bar did not (first breakout only)
        if close > asian_high and previous_bar["close"] <= asian_high:
            direction = "BUY"
        elif close < asian_low and previous_bar["close"] >= asian_low:
            direction = "SELL"

        if direction is None:
            log.info(f"No breakout — close: {close:.2f} | Asian High: {asian_high:.2f} | Asian Low: {asian_low:.2f}")
            return None

        log.info(f"Breakout detected — {direction} | close: {close:.2f}")

        # --- Step 3: EMA 55 filter ----------------------------------------
        ema55 = self._ema(bars_15m["close"], 55).iloc[-1]

        if direction == "BUY" and close <= ema55:
            log.info(f"BUY filtered — close {close:.2f} below EMA55 {ema55:.2f}")
            return None

        if direction == "SELL" and close >= ema55:
            log.info(f"SELL filtered — close {close:.2f} above EMA55 {ema55:.2f}")
            return None

        log.info(f"EMA55 filter passed — EMA55: {ema55:.2f}")

        # --- Step 4: ATR-based SL and TP ----------------------------------
        atr = self._atr(bars_15m, period=14).iloc[-1]

        tick        = mt5.symbol_info_tick(config.SYMBOL)
        entry_price = tick.ask if direction == "BUY" else tick.bid

        if direction == "BUY":
            stop_loss   = round(entry_price - (1.5 * atr), 2)
            take_profit = round(entry_price + (2.0 * atr), 2)
        else:
            stop_loss   = round(entry_price + (1.5 * atr), 2)
            take_profit = round(entry_price - (2.0 * atr), 2)

        # Minimum SL distance of 8 points
        sl_dist = abs(entry_price - stop_loss)
        if sl_dist < 8.0:
            if direction == "BUY":
                stop_loss = round(entry_price - 8.0, 2)
            else:
                stop_loss = round(entry_price + 8.0, 2)
            log.info(f"SL widened to minimum 8 points — SL: {stop_loss:.2f}")

        log.info(f"ATR: {atr:.2f} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        # --- Step 5: Confidence score ------------------------------------
        confidence = 0.65  # base — breakout + EMA filter

        # +0.10 if breakout candle volume > 20-bar average
        avg_volume = bars_15m["tick_volume"].iloc[-21:-1].mean()
        if current_bar["tick_volume"] > avg_volume:
            confidence += 0.10
            log.info(f"Volume boost +0.10 — bar vol: {current_bar['tick_volume']:.0f} > avg: {avg_volume:.0f}")

        # +0.05 if 1h EMA 21 agrees with direction
        bars_1h = self._get_bars(mt5.TIMEFRAME_H1, count=50)
        if bars_1h is not None and len(bars_1h) >= 22:
            ema21_1h   = self._ema(bars_1h["close"], 21).iloc[-1]
            close_1h   = bars_1h["close"].iloc[-1]
            if direction == "BUY" and close_1h > ema21_1h:
                confidence += 0.05
                log.info(f"1h EMA21 agrees +0.05 — close: {close_1h:.2f} > EMA21: {ema21_1h:.2f}")
            elif direction == "SELL" and close_1h < ema21_1h:
                confidence += 0.05
                log.info(f"1h EMA21 agrees +0.05 — close: {close_1h:.2f} < EMA21: {ema21_1h:.2f}")

        confidence = round(min(confidence, 1.0), 2)
        log.info(f"Final confidence: {confidence}")

        if confidence < self.MIN_CONFIDENCE:
            log.info(f"Confidence {confidence} below minimum {self.MIN_CONFIDENCE} — no signal")
            return None

        # --- Step 6: Build signal dict ------------------------------------
        self._signal_count += 1
        signal_id = f"LB_{now_utc.strftime('%Y%m%d')}_{self._signal_count:03d}"

        signal = {
            "symbol":      config.SYMBOL,
            "direction":   direction,
            "entry_price": entry_price,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "confidence":  confidence,
            "signal_id":   signal_id,
            "atr":         round(atr, 2),
            "asian_high":  asian_high,
            "asian_low":   asian_low,
            "ema55":       round(ema55, 2),
        }

        self._traded_today = True
        log.info(f"Signal generated: {signal}")
        return signal

    # -------------------------------------------------------------------------
    # ASIAN RANGE
    # -------------------------------------------------------------------------

    def _calc_asian_range(self, bars: pd.DataFrame, now_utc: datetime):
        """
        Find all 15m bars from 00:00 UTC to 07:00 UTC today.
        Return (high, low, range) or (None, None, None).
        """
        asian_start = now_utc.replace(hour=0,  minute=0, second=0, microsecond=0, tzinfo=None)
        asian_end   = now_utc.replace(hour=7,  minute=0, second=0, microsecond=0, tzinfo=None)

        # Filter bars within Asian session
        asian_bars = bars[
            (bars["time"] >= asian_start) &
            (bars["time"] <  asian_end)
        ]

        if len(asian_bars) < 5:
            return None, None, None

        high  = asian_bars["high"].max()
        low   = asian_bars["low"].min()
        rng   = round(high - low, 2)
        return round(high, 2), round(low, 2), rng

    # -------------------------------------------------------------------------
    # SESSION WINDOW CHECK
    # -------------------------------------------------------------------------

    def _in_london_window(self, now_utc: datetime) -> bool:
        """Return True if current time is within the London breakout window."""
        session_start_h = int(config.ASIAN_SESSION_END_UTC.split(":")[0])
        session_end_h   = int(config.LONDON_SESSION_END_UTC.split(":")[0])
        return session_start_h <= now_utc.hour < session_end_h

    # -------------------------------------------------------------------------
    # DATA FETCHING
    # -------------------------------------------------------------------------

    def _get_bars(self, timeframe: int, count: int = 200) -> pd.DataFrame | None:
        """Fetch OHLCV bars from MT5 and return as DataFrame with UTC times."""
        rates = mt5.copy_rates_from_pos(config.SYMBOL, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            log.error(f"No bars returned from MT5 — {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["time"] = pd.to_datetime(df["time"])

        # Make timezone-naive for easier comparison
        df["time"] = df["time"].apply(lambda x: x.replace(tzinfo=None))
        return df

    # -------------------------------------------------------------------------
    # INDICATORS
    # -------------------------------------------------------------------------

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _atr(self, bars: pd.DataFrame, period: int = 14) -> pd.Series:
        high  = bars["high"]
        low   = bars["low"]
        close = bars["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()