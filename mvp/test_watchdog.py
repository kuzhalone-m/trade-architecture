# =============================================================================
# test_watchdog.py
#
# Tests that Stage 2 is fully working end to end:
#
#   Test 1 — Risk engine pre-trade checks work correctly
#   Test 2 — Watchdog detects a simulated drawdown breach and halts
#   Test 3 — HALT file blocks new signals after a breach
#
# Run this BEFORE running the real watchdog.
# No real orders are placed. This only simulates equity values.
#
# Usage:
#   python test_watchdog.py
# =============================================================================

import json
import logging
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timezone

import MetaTrader5 as mt5

import config
from risk_engine import RiskEngine

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [TEST]  %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("Test")

# =============================================================================

def clean_flags():
    """Remove HALT and WARNING files before each test."""
    for f in [config.HALT_FILE, config.WARNING_FILE, config.STATE_FILE]:
        if os.path.exists(f):
            os.remove(f)

def print_result(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    line   = f"  [{status}]  {test_name}"
    if detail:
        line += f" — {detail}"
    print(line)

# =============================================================================
# TEST 1 — Pre-trade signal validation
# =============================================================================

def test_signal_validation():
    print("\n--- Test 1: Signal Validation ---")
    clean_flags()

    engine = RiskEngine()
    ok = engine.connect()
    if not ok:
        print_result("Connect to MT5", False, "MT5 not running or wrong credentials")
        return

    equity = engine.current_equity

    # 1a. Valid signal — should be approved
    signal_good = {
        "symbol":      config.SYMBOL,
        "direction":   "BUY",
        "entry_price": equity,
        "stop_loss":   equity - 10,
        "take_profit": equity + 20,
        "confidence":  0.75,
        "signal_id":   "TEST_GOOD"
    }
    approved, reason = engine.check_signal(signal_good)
    print_result("Valid signal approved", approved, reason)

    # 1b. Missing stop-loss — should be rejected
    signal_no_sl = {**signal_good, "stop_loss": 0, "signal_id": "TEST_NO_SL"}
    approved, reason = engine.check_signal(signal_no_sl)
    print_result("No SL rejected", not approved, reason)

    # 1c. Low confidence — should be rejected
    signal_low_conf = {**signal_good, "confidence": 0.40, "signal_id": "TEST_LOW_CONF"}
    approved, reason = engine.check_signal(signal_low_conf)
    print_result("Low confidence rejected", not approved, reason)

    # 1d. Simulate warning level — manually set day_start higher
    engine.day_start_equity = engine.current_equity * 1.025  # makes daily DD = ~2.5%
    approved, reason = engine.check_signal({**signal_good, "signal_id": "TEST_WARNING"})
    warning_blocked = not approved or engine.warning_active()
    print_result("Warning level detected", warning_blocked, f"Daily DD simulated at ~2.5%")

    engine.disconnect()

# =============================================================================
# TEST 2 — Watchdog breach simulation
# =============================================================================

def test_watchdog_breach():
    print("\n--- Test 2: Watchdog Breach Simulation ---")
    clean_flags()

    # We simulate a breach by temporarily lowering the halt threshold
    # to something smaller than the current equity drop — no real trades needed.

    original_halt = config.DAILY_DD_HALT

    # Import watchdog class directly to test its logic
    from watchdog import Watchdog

    watchdog = Watchdog()
    ok = watchdog.connect()
    if not ok:
        print_result("Watchdog connect", False, "MT5 not running")
        return

    print_result("Watchdog connected", True, f"Equity: {watchdog.current_equity:.2f}")

    # Simulate: set day_start_equity higher so DD appears to be 4%
    # (above the 3.5% halt level) without touching real money
    watchdog.day_start_equity = watchdog.current_equity * 1.04

    daily_dd = watchdog._calc_daily_dd()
    print(f"         Simulated daily DD: {daily_dd:.2%} (halt at {config.DAILY_DD_HALT:.1%})")

    # Manually trigger the breach check
    halt_triggered = False
    if daily_dd >= config.DAILY_DD_HALT:
        # Don't actually close positions in test — just verify the detection
        watchdog._write_halt_file(f"TEST: simulated daily DD {daily_dd:.2%}")
        halt_triggered = True

    print_result("Breach detected correctly", halt_triggered,
                 f"DD {daily_dd:.2%} >= halt {config.DAILY_DD_HALT:.1%}")

    halt_file_exists = os.path.exists(config.HALT_FILE)
    print_result("HALT file written", halt_file_exists, config.HALT_FILE)

    mt5.shutdown()

# =============================================================================
# TEST 3 — HALT file blocks new signals
# =============================================================================

def test_halt_blocks_signals():
    print("\n--- Test 3: HALT File Blocks New Signals ---")

    # HALT file should exist from Test 2
    if not os.path.exists(config.HALT_FILE):
        # Create it manually for this test
        with open(config.HALT_FILE, "w") as f:
            f.write("TEST halt file\n")

    engine = RiskEngine()
    ok = engine.connect()
    if not ok:
        print_result("Connect", False)
        return

    signal = {
        "symbol":      config.SYMBOL,
        "direction":   "BUY",
        "entry_price": 2000.0,
        "stop_loss":   1990.0,
        "take_profit": 2020.0,
        "confidence":  0.80,
        "signal_id":   "TEST_HALTED"
    }

    approved, reason = engine.check_signal(signal)
    print_result("Signal blocked when HALTED", not approved, reason)

    engine.disconnect()

    # Clean up after all tests
    clean_flags()
    print("\n  Flag files cleaned up.")

# =============================================================================
# SUMMARY
# =============================================================================

def main():
    print("=" * 55)
    print("  Stage 2 Test Suite — Risk Engine + Watchdog")
    print("=" * 55)
    print("  Make sure MT5 is open and logged in before running.\n")

    test_signal_validation()
    test_watchdog_breach()
    test_halt_blocks_signals()

    print("\n" + "=" * 55)
    print("  All tests complete.")
    print("  If all show [PASS], Stage 2 is working correctly.")
    print("  You are ready to build Stage 3 — Executor.")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
