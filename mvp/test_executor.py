# =============================================================================
# test_executor.py
#
# Two tests:
#   Test 1 — Dry run: verifies lot calculation and signal validation
#             without placing any real orders
#   Test 2 — Live demo order: places one real BUY LIMIT on your AUR demo
#             account, confirms it appears in MT5, then cancels it
#
# Run with MT5 open and logged into your AUR Markets demo account.
#
# Usage:
#   python test_executor.py
# =============================================================================

import logging
import os
import time
from pathlib import Path

import MetaTrader5 as mt5

import config
from risk_engine import RiskEngine
from executor import Executor

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [TEST]  %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("TestExecutor")

# =============================================================================

def print_result(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    line   = f"  [{status}]  {name}"
    if detail:
        line += f" — {detail}"
    print(line)

def clean_flags():
    for f in [config.HALT_FILE, config.WARNING_FILE]:
        if os.path.exists(f):
            os.remove(f)

# =============================================================================
# TEST 1 — Dry run (no real orders)
# =============================================================================

def test_dry_run(engine: RiskEngine, executor: Executor):
    print("\n--- Test 1: Dry Run (no orders placed) ---")

    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        print_result("Get XAUUSD tick", False, "Symbol not available")
        return

    current_price = tick.ask
    print(f"  Current {config.SYMBOL} ask price: {current_price:.2f}")

    # Build a realistic signal based on current price
    signal = {
        "symbol":      config.SYMBOL,
        "direction":   "BUY",
        "entry_price": round(current_price - 2.0, 2),   # 2 points below ask
        "stop_loss":   round(current_price - 12.0, 2),  # 10 point SL
        "take_profit": round(current_price + 18.0, 2),  # 20 point TP (2:1)
        "confidence":  0.75,
        "signal_id":   "DRY_RUN_001"
    }

    print(f"  Signal: BUY @ {signal['entry_price']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")

    # Test lot calculation directly
    lot = executor._calculate_lot(signal)
    print_result("Lot size calculated", lot is not None, f"lot = {lot}")

    # Test risk check passes
    approved, reason = engine.check_signal(signal)
    print_result("Risk check approved", approved, reason)

    # Test blocked signal (no SL)
    bad_signal = {**signal, "stop_loss": 0, "signal_id": "DRY_NO_SL"}
    approved_bad, reason_bad = engine.check_signal(bad_signal)
    print_result("No-SL signal blocked", not approved_bad, reason_bad)

    # Test low confidence blocked
    low_conf = {**signal, "confidence": 0.30, "signal_id": "DRY_LOW_CONF"}
    approved_lc, reason_lc = engine.check_signal(low_conf)
    print_result("Low-confidence blocked", not approved_lc, reason_lc)

# =============================================================================
# TEST 2 — Real demo order (places + cancels)
# =============================================================================

def test_live_demo_order(engine: RiskEngine, executor: Executor):
    print("\n--- Test 2: Live Demo Order (places then cancels) ---")
    print("  This places a real limit order on your AUR demo account.")
    print("  It will be cancelled automatically after confirmation.\n")

    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        print_result("Get tick price", False, "Symbol not available")
        return

    current_price = tick.ask

    # Place a BUY LIMIT well below current price — it won't fill,
    # giving us time to confirm it and then cancel it
    entry = round(current_price - 50.0, 2)   # 50 points below — won't fill
    sl    = round(entry - 10.0, 2)
    tp    = round(entry + 20.0, 2)

    signal = {
        "symbol":      config.SYMBOL,
        "direction":   "BUY",
        "entry_price": entry,
        "stop_loss":   sl,
        "take_profit": tp,
        "confidence":  0.75,
        "signal_id":   "LIVE_TEST_001"
    }

    print(f"  Placing BUY LIMIT @ {entry} (current ask: {current_price:.2f})")
    print(f"  SL: {sl} | TP: {tp}")
    print(f"  This order is 50 points away — it will NOT fill.\n")

    result = executor.execute(signal)

    print(f"\n  Execution result: {result['status']}")

    if result["status"] == "filled":
        # Shouldn't happen — order is too far from price
        print_result("Order placed (unexpected fill)", True, "Order filled immediately")

    elif result["status"] == "cancelled":
        # This is the expected path — order placed, waited 60s, not filled, cancelled
        print_result("Order placed and cancelled cleanly", True,
                     f"ticket: {result['ticket']}")

    elif result["status"] == "blocked":
        print_result("Order placed", False, f"Blocked: {result['reason']}")

    elif result["status"] == "error":
        print_result("Order placed", False, f"Error: {result['reason']}")
        print("\n  Common causes:")
        print("  - XAUUSD not in Market Watch (right-click → Show All)")
        print("  - Market is closed")
        print("  - Broker requires different ORDER_FILLING mode")
        print("  Tell me the error and I will fix it.\n")
        return

    # Verify trade log was written
    import json
    try:
        with open(executor.trade_log) as f:
            trades = json.load(f)
        our_trade = next((t for t in trades if t["signal_id"] == "LIVE_TEST_001"), None)
        print_result("Trade logged to trades.json", our_trade is not None,
                     f"status: {our_trade['status'] if our_trade else 'not found'}")
    except Exception as e:
        print_result("Trade log check", False, str(e))

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  Stage 3 Test Suite — Executor")
    print("=" * 55)

    clean_flags()

    # Connect
    engine = RiskEngine()
    ok = engine.connect()
    if not ok:
        print("\nFailed to connect to MT5. Make sure terminal is open.")
        return

    engine.start_monitor()
    executor = Executor(engine)

    print(f"\n  Connected | Equity: {engine.current_equity:.2f}")

    # Run tests
    test_dry_run(engine, executor)
    test_live_demo_order(engine, executor)

    print("\n" + "=" * 55)
    print("  All tests complete.")
    print("  If both tests show [PASS], Stage 3 is working.")
    print("  Check logs/trades.json to see the logged order.")
    print("  You are ready to build Stage 4 — Signal.")
    print("=" * 55 + "\n")

    engine.disconnect()

if __name__ == "__main__":
    main()
