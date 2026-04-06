# =============================================================================
# test_signal.py
#
# Three tests:
#   Test 1 — Data fetch: confirms MT5 is returning XAUUSD bars correctly
#   Test 2 — Indicator check: verifies EMA55 and ATR are calculating correctly
#   Test 3 — Historical replay: runs the strategy logic on the last 30 days
#            of 15m data and prints every signal it would have generated
#
# This does NOT place any orders. Pure logic verification.
#
# Usage:
#   python test_signal.py
# =============================================================================

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import config
from risk_engine import RiskEngine
from strategy import SignalEngine

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,   # Suppress info logs during replay
    format="%(asctime)s  [TEST]  %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("TestSignal")


def print_result(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    line   = f"  [{status}]  {name}"
    if detail:
        line += f" — {detail}"
    print(line)

# =============================================================================
# TEST 1 — Data fetch
# =============================================================================

def test_data_fetch(engine: SignalEngine):
    print("\n--- Test 1: Data Fetch ---")

    bars_15m = engine._get_bars(mt5.TIMEFRAME_M15, count=200)
    ok_15m   = bars_15m is not None and len(bars_15m) >= 100
    print_result(
        "200 x 15m bars fetched",
        ok_15m,
        f"{len(bars_15m)} bars, latest: {bars_15m['time'].iloc[-1]}" if ok_15m else "failed"
    )

    bars_1h = engine._get_bars(mt5.TIMEFRAME_H1, count=50)
    ok_1h   = bars_1h is not None and len(bars_1h) >= 30
    print_result(
        "50 x 1h bars fetched",
        ok_1h,
        f"{len(bars_1h)} bars" if ok_1h else "failed"
    )

    if ok_15m:
        latest = bars_15m.iloc[-1]
        print(f"\n  Latest 15m bar:")
        print(f"    Time:   {latest['time']}")
        print(f"    Open:   {latest['open']:.2f}")
        print(f"    High:   {latest['high']:.2f}")
        print(f"    Low:    {latest['low']:.2f}")
        print(f"    Close:  {latest['close']:.2f}")
        print(f"    Volume: {latest['tick_volume']:.0f}")

    return bars_15m, bars_1h

# =============================================================================
# TEST 2 — Indicator check
# =============================================================================

def test_indicators(engine: SignalEngine, bars_15m: pd.DataFrame):
    print("\n--- Test 2: Indicator Check ---")

    if bars_15m is None:
        print("  Skipped — no bar data")
        return

    ema55 = engine._ema(bars_15m["close"], 55)
    atr14 = engine._atr(bars_15m, 14)

    ema_ok = not ema55.isna().all() and ema55.iloc[-1] > 0
    atr_ok = not atr14.isna().all() and atr14.iloc[-1] > 0

    print_result("EMA 55 calculated", ema_ok, f"current: {ema55.iloc[-1]:.2f}")
    print_result("ATR 14 calculated", atr_ok, f"current: {atr14.iloc[-1]:.2f}")

    close = bars_15m["close"].iloc[-1]
    print(f"\n  Current close vs EMA55: {close:.2f} vs {ema55.iloc[-1]:.2f}")
    print(f"  Close is {'ABOVE' if close > ema55.iloc[-1] else 'BELOW'} EMA55")
    print(f"  ATR 14: {atr14.iloc[-1]:.2f} points")
    print(f"  Implied SL (1.5x ATR): {atr14.iloc[-1] * 1.5:.2f} points")
    print(f"  Implied TP (2.0x ATR): {atr14.iloc[-1] * 2.0:.2f} points")

# =============================================================================
# TEST 3 — Historical replay (last 30 days)
# =============================================================================

def test_historical_replay(engine: SignalEngine):
    print("\n--- Test 3: Historical Replay (last 30 trading days) ---")
    print("  Replaying strategy logic day by day...\n")

    # Fetch 30 days of 15m bars (30 days * 24h * 4 bars/h = ~2880 bars)
    rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M15, 0, 3000)
    if rates is None or len(rates) < 100:
        print("  Could not fetch historical data")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Pre-calculate indicators on full dataset
    df["ema55"] = df["close"].ewm(span=55, adjust=False).mean()
    df["atr14"] = _atr_series(df, 14)
    df["vol_avg20"] = df["tick_volume"].rolling(20).mean()

    signals_found  = []
    days_checked   = 0
    days_no_signal = 0

    # Group by date and replay each day
    df["date"] = df["time"].dt.date
    unique_dates = sorted(df["date"].unique())[-30:]  # last 30 days

    for date in unique_dates:
        days_checked += 1
        day_bars = df[df["date"] == date].copy()

        # Asian range: bars from 00:00 to 07:00
        asian = day_bars[day_bars["time"].dt.hour < 7]
        if len(asian) < 5:
            continue

        asian_high  = asian["high"].max()
        asian_low   = asian["low"].min()
        asian_range = asian_high - asian_low

        if asian_range < SignalEngine.MIN_ASIAN_RANGE:
            continue

        # London window: bars from 07:00 to 10:00
        london = day_bars[
            (day_bars["time"].dt.hour >= 7) &
            (day_bars["time"].dt.hour < 10)
        ]

        signal_taken = False
        for i in range(1, len(london)):
            bar      = london.iloc[i]
            prev_bar = london.iloc[i - 1]
            close    = bar["close"]
            ema55    = bar["ema55"]
            atr      = bar["atr14"]

            # Skip if indicators not ready
            if pd.isna(ema55) or pd.isna(atr) or atr == 0:
                continue

            direction = None
            if close > asian_high and prev_bar["close"] <= asian_high:
                direction = "BUY"
            elif close < asian_low and prev_bar["close"] >= asian_low:
                direction = "SELL"

            if direction is None:
                continue

            # EMA filter
            if direction == "BUY" and close <= ema55:
                continue
            if direction == "SELL" and close >= ema55:
                continue

            # Confidence
            confidence = 0.65
            if bar["tick_volume"] > bar["vol_avg20"]:
                confidence += 0.10

            # SL / TP
            if direction == "BUY":
                sl = close - (1.5 * atr)
                tp = close + (2.0 * atr)
            else:
                sl = close + (1.5 * atr)
                tp = close - (2.0 * atr)

            rr = round(abs(tp - close) / abs(close - sl), 2)

            signals_found.append({
                "date":       str(date),
                "time":       str(bar["time"]),
                "direction":  direction,
                "entry":      round(close, 2),
                "sl":         round(sl, 2),
                "tp":         round(tp, 2),
                "atr":        round(atr, 2),
                "confidence": confidence,
                "rr":         rr,
                "asian_high": round(asian_high, 2),
                "asian_low":  round(asian_low, 2),
            })
            signal_taken = True
            break   # one signal per day

        if not signal_taken:
            days_no_signal += 1

    # Print results
    print(f"  Days checked:       {days_checked}")
    print(f"  Signals generated:  {len(signals_found)}")
    print(f"  Days no signal:     {days_no_signal}")
    print(f"  Signal frequency:   {len(signals_found)}/{days_checked} days")

    if signals_found:
        print(f"\n  {'Date':<12} {'Dir':<5} {'Entry':>8} {'SL':>8} {'TP':>8} {'ATR':>6} {'Conf':>5} {'RR':>4}")
        print("  " + "-" * 62)
        for s in signals_found[-15:]:   # show last 15
            print(
                f"  {s['date']:<12} {s['direction']:<5} "
                f"{s['entry']:>8.2f} {s['sl']:>8.2f} {s['tp']:>8.2f} "
                f"{s['atr']:>6.2f} {s['confidence']:>5.2f} {s['rr']:>4.1f}"
            )

        avg_atr  = round(sum(s["atr"] for s in signals_found) / len(signals_found), 2)
        avg_conf = round(sum(s["confidence"] for s in signals_found) / len(signals_found), 2)
        buys     = sum(1 for s in signals_found if s["direction"] == "BUY")
        sells    = len(signals_found) - buys

        print(f"\n  Average ATR:        {avg_atr}")
        print(f"  Average confidence: {avg_conf}")
        print(f"  BUY signals:        {buys}")
        print(f"  SELL signals:       {sells}")

        print_result(
            "Historical replay complete",
            len(signals_found) > 0,
            f"{len(signals_found)} signals found in {days_checked} days"
        )
    else:
        print("\n  No signals found in replay period.")
        print("  This could mean:")
        print("  - Market was closed or data is sparse")
        print("  - Asian range was consistently too tight")
        print("  - EMA filter blocked all breakouts")
        print_result("Historical replay", False, "0 signals found")

# =============================================================================
# HELPER — ATR for replay (standalone, no class dependency)
# =============================================================================

def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  Stage 4 Test Suite — Signal Engine")
    print("=" * 55)

    # Connect MT5
    if config.MT5_LOGIN:
        ok = mt5.initialize(
            login=config.MT5_LOGIN,
            password=config.MT5_PASSWORD,
            server=config.MT5_SERVER
        )
    else:
        ok = mt5.initialize()

    if not ok:
        print("Failed to connect to MT5.")
        return

    engine = SignalEngine()

    bars_15m, bars_1h = test_data_fetch(engine)
    test_indicators(engine, bars_15m)
    test_historical_replay(engine)

    # Live check — will only generate a signal if currently in London window
    print("\n--- Live Check ---")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    print(f"  Current UTC time: {now.strftime('%H:%M')}")
    if engine._in_london_window(now):
        print("  Inside London window — running live check...")
        signal = engine.check()
        if signal:
            print(f"  LIVE SIGNAL: {signal}")
        else:
            print("  No signal right now — conditions not met")
    else:
        print(f"  Outside London window ({config.ASIAN_SESSION_END_UTC}–{config.LONDON_SESSION_END_UTC} UTC) — live check skipped")
        print("  Run this between 07:00–10:00 UTC to test live signal generation")

    print("\n" + "=" * 55)
    print("  All tests complete.")
    print("  Review the historical replay above.")
    print("  If signals look reasonable, Stage 4 is working.")
    print("  You are ready to build Stage 5 — Backtester.")
    print("=" * 55 + "\n")

    mt5.shutdown()

if __name__ == "__main__":
    main()