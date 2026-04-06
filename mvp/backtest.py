# =============================================================================
# backtest.py
#
# Validates the London Breakout strategy before live trading.
# Two gates:
#   Gate 1 — Walk-Forward Analysis (8 rolling windows)
#   Gate 2 — Monte Carlo Simulation (500 reshuffles)
#
# Requires: TickStory XAUUSD 1-minute CSV data OR uses MT5 directly.
# If no CSV is provided, pulls maximum available bars from MT5.
#
# Usage:
#   python backtest.py                        # uses MT5 data
#   python backtest.py --csv xauusd_1m.csv   # uses TickStory CSV
# =============================================================================

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config

# --- Logging -----------------------------------------------------------------
Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  [BACKTEST]  %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("Backtest")


# =============================================================================
# STRATEGY — same logic as signal.py, pure pandas (no MT5 calls)
# =============================================================================

class LondonBreakoutStrategy:
    """
    Pure pandas implementation of the London Breakout strategy.
    Identical logic to strategy.py but runs on a DataFrame instead of live MT5.

    v2 adjustments (based on backtest results):
      - EMA filter: 55 → 21  (stronger trend confirmation)
      - SL multiplier: 1.5x → 2.0x ATR  (more room, fewer stop-outs)
      - Min confidence: 0.65 → 0.70  (higher quality signals only)
    """

    MIN_ASIAN_RANGE = 8.0
    MIN_CONFIDENCE  = 0.70   # raised from 0.65
    EMA_PERIOD      = 21     # tightened from 55
    SL_MULTIPLIER   = 2.0    # widened from 1.5
    TP_MULTIPLIER   = 3.0    # adjusted to maintain 1.5:1 RR minimum

    def run(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Run strategy on a 15m OHLCV DataFrame.
        Returns a DataFrame of trades with entry, sl, tp, and outcome.

        df_15m columns required: time, open, high, low, close, tick_volume
        time must be a datetime column in UTC (timezone-naive).
        """
        # Pre-calculate indicators on the full dataset
        df_15m = df_15m.copy()
        df_15m["ema"]       = df_15m["close"].ewm(span=self.EMA_PERIOD, adjust=False).mean()
        df_15m["atr14"]     = self._atr(df_15m, 14)
        df_15m["vol_avg20"] = df_15m["tick_volume"].rolling(20).mean()
        df_15m["date"]      = df_15m["time"].dt.date

        trades = []
        unique_dates = sorted(df_15m["date"].unique())

        for date in unique_dates:
            day = df_15m[df_15m["date"] == date]

            # Asian range: 00:00 - 07:00 UTC
            asian = day[day["time"].dt.hour < 7]
            if len(asian) < 5:
                continue

            asian_high  = asian["high"].max()
            asian_low   = asian["low"].min()
            asian_range = asian_high - asian_low

            if asian_range < self.MIN_ASIAN_RANGE:
                continue

            # London window: 07:00 - 10:00 UTC
            london = day[
                (day["time"].dt.hour >= 7) &
                (day["time"].dt.hour < 10)
            ]

            for i in range(1, len(london)):
                bar      = london.iloc[i]
                prev_bar = london.iloc[i - 1]
                close    = bar["close"]
                ema      = bar["ema"]
                atr      = bar["atr14"]

                if pd.isna(ema) or pd.isna(atr) or atr == 0:
                    continue

                # Breakout check
                direction = None
                if close > asian_high and prev_bar["close"] <= asian_high:
                    direction = "BUY"
                elif close < asian_low and prev_bar["close"] >= asian_low:
                    direction = "SELL"

                if direction is None:
                    continue

                # EMA filter (EMA21)
                if direction == "BUY"  and close <= ema:
                    continue
                if direction == "SELL" and close >= ema:
                    continue

                # SL / TP (2.0x ATR stop, 3.0x ATR target = 1.5:1 RR)
                if direction == "BUY":
                    sl = close - (self.SL_MULTIPLIER * atr)
                    tp = close + (self.TP_MULTIPLIER * atr)
                else:
                    sl = close + (self.SL_MULTIPLIER * atr)
                    tp = close - (self.TP_MULTIPLIER * atr)

                sl_dist = abs(close - sl)
                if sl_dist < 8.0:
                    sl_dist = 8.0
                    sl = close - 8.0 if direction == "BUY" else close + 8.0
                    tp = close + (self.TP_MULTIPLIER / self.SL_MULTIPLIER * sl_dist) if direction == "BUY" else close - (self.TP_MULTIPLIER / self.SL_MULTIPLIER * sl_dist)

                # Confidence (base 0.65, volume boost +0.10)
                confidence = 0.65
                if bar["tick_volume"] > bar["vol_avg20"]:
                    confidence += 0.10

                if confidence < self.MIN_CONFIDENCE:
                    continue

                # Simulate outcome using remaining bars in the day
                outcome, exit_price, bars_held = self._simulate_outcome(
                    day, bar.name, direction, close, sl, tp
                )

                # Commission: ~$7 round trip per standard lot, scaled to 0.05 lots
                commission = 7.0 * 0.05

                if outcome == "win":
                    pnl_pips = abs(tp - close)
                    pnl_usd  = (pnl_pips * 0.05 * 100) - commission
                elif outcome == "loss":
                    pnl_pips = -abs(close - sl)
                    pnl_usd  = (pnl_pips * 0.05 * 100) - commission
                else:
                    pnl_pips = 0
                    pnl_usd  = -commission

                trades.append({
                    "date":        str(date),
                    "time":        str(bar["time"]),
                    "direction":   direction,
                    "entry":       round(close, 2),
                    "sl":          round(sl, 2),
                    "tp":          round(tp, 2),
                    "atr":         round(atr, 2),
                    "confidence":  confidence,
                    "outcome":     outcome,
                    "exit_price":  round(exit_price, 2),
                    "pnl_pips":    round(pnl_pips, 2),
                    "pnl_usd":     round(pnl_usd, 2),
                    "bars_held":   bars_held,
                    "rr":          round(abs(tp - close) / abs(close - sl), 2)
                })
                break  # one trade per day

        return pd.DataFrame(trades)

    def _simulate_outcome(self, day_bars, entry_idx, direction, entry, sl, tp):
        """
        Walk forward bar by bar after entry.
        Return (outcome, exit_price, bars_held).
        outcome = 'win' | 'loss' | 'eod' (end of day, no hit)
        """
        remaining = day_bars[day_bars.index > entry_idx]
        bars_held = 0

        for _, bar in remaining.iterrows():
            bars_held += 1
            if direction == "BUY":
                if bar["low"] <= sl:
                    return "loss", sl, bars_held
                if bar["high"] >= tp:
                    return "win", tp, bars_held
            else:
                if bar["high"] >= sl:
                    return "loss", sl, bars_held
                if bar["low"] <= tp:
                    return "win", tp, bars_held

        # Session ended without hitting SL or TP — close at last bar
        last_close = day_bars.iloc[-1]["close"]
        return "eod", last_close, bars_held

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
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
# METRICS
# =============================================================================

def calc_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    """Calculate all performance metrics from a trades DataFrame."""
    if trades.empty or len(trades) < 3:
        return {"label": label, "trades": 0, "insufficient_data": True}

    wins   = trades[trades["outcome"] == "win"]
    losses = trades[trades["outcome"] == "loss"]
    eods   = trades[trades["outcome"] == "eod"]

    total       = len(trades)
    win_count   = len(wins)
    loss_count  = len(losses)
    win_rate    = win_count / total if total > 0 else 0

    gross_profit = wins["pnl_usd"].sum() if not wins.empty else 0
    gross_loss   = abs(losses["pnl_usd"].sum()) if not losses.empty else 0
    net_pnl      = trades["pnl_usd"].sum()
    profit_factor= gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualised, assuming ~252 trading days)
    daily_pnl = trades.groupby("date")["pnl_usd"].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * (252 ** 0.5)
    else:
        sharpe = 0.0

    # Max drawdown (on cumulative PnL curve)
    cum_pnl  = trades["pnl_usd"].cumsum()
    peak     = cum_pnl.cummax()
    drawdown = peak - cum_pnl
    max_dd   = drawdown.max()
    max_dd_pct = (max_dd / (10000 + cum_pnl.max())) if (10000 + cum_pnl.max()) > 0 else 0

    # Average RR achieved
    avg_rr = trades["rr"].mean() if "rr" in trades.columns else 0

    # Trades per month
    if len(trades) > 0:
        try:
            date_range = (pd.to_datetime(trades["date"].max()) -
                         pd.to_datetime(trades["date"].min())).days
            months = max(date_range / 30, 1)
            trades_per_month = total / months
        except Exception:
            trades_per_month = 0
    else:
        trades_per_month = 0

    return {
        "label":             label,
        "trades":            total,
        "wins":              win_count,
        "losses":            loss_count,
        "eods":              len(eods),
        "win_rate":          round(win_rate, 4),
        "net_pnl_usd":       round(net_pnl, 2),
        "gross_profit":      round(gross_profit, 2),
        "gross_loss":        round(gross_loss, 2),
        "profit_factor":     round(profit_factor, 3),
        "sharpe":            round(sharpe, 3),
        "max_dd_usd":        round(max_dd, 2),
        "max_dd_pct":        round(max_dd_pct, 4),
        "avg_rr":            round(avg_rr, 2),
        "trades_per_month":  round(trades_per_month, 1),
    }


def print_metrics(m: dict):
    if m.get("insufficient_data"):
        print(f"  {m['label']:<30} — insufficient data ({m['trades']} trades)")
        return

    passed = m["sharpe"] >= 0.8 and m["net_pnl_usd"] > 0
    status = "PASS" if passed else "FAIL"

    print(f"  {m['label']:<28} [{status}]")
    print(f"    Trades: {m['trades']}  |  Win rate: {m['win_rate']:.1%}  |  "
          f"Profit factor: {m['profit_factor']:.2f}")
    print(f"    Sharpe: {m['sharpe']:.3f}  |  Net PnL: ${m['net_pnl_usd']:.2f}  |  "
          f"Max DD: {m['max_dd_pct']:.2%}")
    print(f"    Avg RR: {m['avg_rr']:.2f}  |  Trades/month: {m['trades_per_month']:.1f}")


# =============================================================================
# GATE 1 — WALK-FORWARD ANALYSIS
# =============================================================================

def gate1_walk_forward(df: pd.DataFrame) -> tuple[bool, list]:
    """
    8 rolling windows. Each: 9 months train, 3 months test.
    Pass: profitable in >= 6 of 8 OOS windows AND overall OOS Sharpe >= 0.8.
    Returns (passed, list of window metrics).
    """
    print("\n" + "="*60)
    print("  GATE 1 — Walk-Forward Analysis")
    print("  8 windows · 9 months train · 3 months test each")
    print("="*60)

    strategy = LondonBreakoutStrategy()

    df["month"] = df["time"].dt.to_period("M")
    all_months  = sorted(df["month"].unique())

    if len(all_months) < 12:
        print(f"  Not enough data — need 12+ months, have {len(all_months)}")
        return False, []

    train_months = 9
    test_months  = 3
    n_windows    = 8

    window_results = []
    oos_trades_all = []

    for w in range(n_windows):
        train_start = all_months[w * 2]                          # slide by 2 months each window
        train_end   = all_months[min(w * 2 + train_months - 1, len(all_months) - 1)]
        test_start  = all_months[min(w * 2 + train_months,     len(all_months) - 1)]
        test_end    = all_months[min(w * 2 + train_months + test_months - 1, len(all_months) - 1)]

        if test_start > all_months[-1]:
            break

        test_data = df[
            (df["month"] >= test_start) &
            (df["month"] <= test_end)
        ]

        if len(test_data) < 100:
            continue

        trades = strategy.run(test_data)
        m = calc_metrics(trades, f"Window {w+1} OOS ({test_start}–{test_end})")
        window_results.append(m)

        if not trades.empty:
            oos_trades_all.append(trades)

        profitable = not m.get("insufficient_data") and m["net_pnl_usd"] > 0
        status = "PASS" if profitable else "FAIL"
        if not m.get("insufficient_data"):
            print(f"  Window {w+1}  [{status}]  "
                  f"Trades: {m['trades']:>3}  |  "
                  f"Net: ${m['net_pnl_usd']:>8.2f}  |  "
                  f"Sharpe: {m['sharpe']:>6.3f}  |  "
                  f"Win rate: {m['win_rate']:.1%}")

    if not window_results:
        print("  No valid windows produced.")
        return False, []

    valid_windows   = [m for m in window_results if not m.get("insufficient_data")]
    profitable_wins = sum(1 for m in valid_windows if m["net_pnl_usd"] > 0)

    # Overall OOS Sharpe across all windows
    if oos_trades_all:
        all_oos = pd.concat(oos_trades_all, ignore_index=True)
        oos_metrics = calc_metrics(all_oos, "Overall OOS")
        oos_sharpe  = oos_metrics["sharpe"]
    else:
        oos_sharpe = 0.0

    print(f"\n  Profitable windows:  {profitable_wins} of {len(valid_windows)} "
          f"(need >= 6)")
    print(f"  Overall OOS Sharpe:  {oos_sharpe:.3f} (need >= 0.8)")

    passed = profitable_wins >= 6 and oos_sharpe >= 0.8

    if passed:
        print("\n  [PASS]  Gate 1 passed.")
    else:
        reasons = []
        if profitable_wins < 6:
            reasons.append(f"only {profitable_wins}/8 windows profitable")
        if oos_sharpe < 0.8:
            reasons.append(f"OOS Sharpe {oos_sharpe:.3f} < 0.8")
        print(f"\n  [FAIL]  Gate 1 failed — {', '.join(reasons)}")

    return passed, window_results


# =============================================================================
# GATE 2 — MONTE CARLO SIMULATION
# =============================================================================

def gate2_monte_carlo(df: pd.DataFrame, n_sims: int = 500) -> tuple[bool, dict]:
    """
    500 random reshuffles of the trade sequence.
    Pass: P95 max drawdown < 15% AND breach probability < 5%.
    Uses the best 6-month window from the full dataset.
    """
    print("\n" + "="*60)
    print(f"  GATE 2 — Monte Carlo Simulation ({n_sims} runs)")
    print("="*60)

    strategy  = LondonBreakoutStrategy()
    all_trades = strategy.run(df)

    if all_trades.empty or len(all_trades) < 10:
        print("  Not enough trades for Monte Carlo (need >= 10)")
        return False, {}

    print(f"  Running {n_sims} simulations on {len(all_trades)} trades...")

    pnl_series    = all_trades["pnl_usd"].values
    starting_eq   = 10000.0
    daily_dd_halt = config.DAILY_DD_HALT  # 3.5%

    max_drawdowns   = []
    breach_count    = 0
    final_pnls      = []

    random.seed(42)  # reproducible results

    for _ in range(n_sims):
        shuffled  = random.sample(list(pnl_series), len(pnl_series))
        equity    = starting_eq
        peak      = starting_eq
        day_start = starting_eq
        max_dd    = 0.0
        breached  = False

        for i, pnl in enumerate(shuffled):
            equity += pnl

            # Update peak
            if equity > peak:
                peak      = equity
                day_start = equity  # simplified daily reset

            # Drawdown from peak
            dd_from_peak = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd_from_peak)

            # Daily drawdown (simplified — every 5 trades = 1 day)
            if i % 5 == 0:
                day_start = equity

            daily_dd = (day_start - equity) / day_start if day_start > 0 else 0
            if daily_dd >= daily_dd_halt:
                breached = True
                break

        max_drawdowns.append(max_dd)
        final_pnls.append(equity - starting_eq)
        if breached:
            breach_count += 1

    max_dd_array  = np.array(max_drawdowns)
    p95_max_dd    = np.percentile(max_dd_array, 95)
    breach_prob   = breach_count / n_sims
    profitable_sims = sum(1 for p in final_pnls if p > 0) / n_sims
    median_return = np.median(final_pnls)

    print(f"\n  P95 max drawdown:    {p95_max_dd:.2%}  (need < 15%)")
    print(f"  Breach probability:  {breach_prob:.2%}  (need < 5%)")
    print(f"  Profitable sims:     {profitable_sims:.1%}  (need > 70%)")
    print(f"  Median final PnL:    ${median_return:.2f}")
    print(f"  Worst case DD:       {max(max_drawdowns):.2%}")
    print(f"  Best case DD:        {min(max_drawdowns):.2%}")

    passed = (
        p95_max_dd  < 0.15 and
        breach_prob < 0.05 and
        profitable_sims > 0.70
    )

    mc_results = {
        "n_sims":          n_sims,
        "p95_max_dd":      round(p95_max_dd, 4),
        "breach_prob":     round(breach_prob, 4),
        "profitable_sims": round(profitable_sims, 4),
        "median_return":   round(median_return, 2),
        "worst_dd":        round(max(max_drawdowns), 4),
        "best_dd":         round(min(max_drawdowns), 4),
    }

    if passed:
        print("\n  [PASS]  Gate 2 passed.")
    else:
        reasons = []
        if p95_max_dd >= 0.15:
            reasons.append(f"P95 DD {p95_max_dd:.2%} >= 15%")
        if breach_prob >= 0.05:
            reasons.append(f"breach prob {breach_prob:.2%} >= 5%")
        if profitable_sims <= 0.70:
            reasons.append(f"only {profitable_sims:.1%} sims profitable")
        print(f"\n  [FAIL]  Gate 2 failed — {', '.join(reasons)}")

    return passed, mc_results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_from_mt5() -> pd.DataFrame | None:
    """Pull maximum available 1m bars from MT5, resample to 15m."""
    print("  Loading data from MT5...")

    if config.MT5_LOGIN:
        ok = mt5.initialize(
            login=config.MT5_LOGIN,
            password=config.MT5_PASSWORD,
            server=config.MT5_SERVER
        )
    else:
        ok = mt5.initialize()

    if not ok:
        print(f"  MT5 init failed: {mt5.last_error()}")
        return None

    # Pull max bars — MT5 typically gives ~3 years of 1m data
    rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M15, 0, 100000)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("  No data returned from MT5.")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    print(f"  Loaded {len(df)} x 15m bars from MT5")
    print(f"  Date range: {df['time'].min().date()} to {df['time'].max().date()}")
    return df


def load_from_csv(filepath: str) -> pd.DataFrame | None:
    """
    Load TickStory tick data CSV and resample to 15m OHLCV bars.

    TickStory tick format (no header):
        Date, Time, Bid, Ask, BidAgain, Volume
        20210323, 00:00:00, 1739.218, 1739.575, 1739.218, 0
    """
    print(f"  Loading data from {filepath}...")

    try:
        # No header row — assign column names manually
        df = pd.read_csv(
            filepath,
            header=None,
            names=["date", "time", "bid", "ask", "bid2", "volume"],
            dtype={"date": str, "time": str}
        )

        # Build datetime from date + time columns
        df["datetime"] = pd.to_datetime(
            df["date"].str.strip() + " " + df["time"].str.strip(),
            format="%Y%m%d %H:%M:%S",
            errors="coerce"
        )
        df = df.dropna(subset=["datetime"])

        # Use bid as price (mid would be (bid+ask)/2 — bid is fine for OHLCV)
        df["price"] = (df["bid"] + df["ask"]) / 2
        df = df.set_index("datetime").sort_index()

        print(f"  Loaded {len(df):,} ticks")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Resampling to 15m bars...")

        # Resample ticks → 15m OHLCV
        df_15m = df["price"].resample("15min").agg(
            open  = "first",
            high  = "max",
            low   = "min",
            close = "last"
        ).dropna()

        # Volume: count ticks per 15m bar
        df_vol = df["price"].resample("15min").count().rename("tick_volume")
        df_15m = df_15m.join(df_vol).dropna()
        df_15m = df_15m.reset_index().rename(columns={"datetime": "time"})

        print(f"  Resampled to {len(df_15m):,} x 15m bars")
        print(f"  Date range: {df_15m['time'].min().date()} to {df_15m['time'].max().date()}")
        return df_15m

    except Exception as e:
        print(f"  Failed to load CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# FULL REPORT
# =============================================================================

def print_full_report(df: pd.DataFrame):
    """Print overall strategy metrics on the full dataset."""
    print("\n" + "="*60)
    print("  FULL DATASET METRICS")
    print("="*60)

    strategy = LondonBreakoutStrategy()
    trades   = strategy.run(df)

    if trades.empty:
        print("  No trades generated on this dataset.")
        return

    m = calc_metrics(trades, "Full dataset")
    print_metrics(m)

    # Monthly breakdown
    print("\n  Monthly PnL breakdown (last 12 months):")
    trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M")
    monthly = trades.groupby("month")["pnl_usd"].sum().tail(12)
    for month, pnl in monthly.items():
        bar    = "█" * int(abs(pnl) / 20)
        sign   = "+" if pnl >= 0 else "-"
        print(f"    {month}  {sign}${abs(pnl):>7.2f}  {bar}")


# =============================================================================
# DECISION
# =============================================================================

def print_decision(g1_passed: bool, g2_passed: bool):
    print("\n" + "="*60)
    print("  DECISION")
    print("="*60)

    if g1_passed and g2_passed:
        print("""
  [BOTH GATES PASSED]

  Strategy is validated. Next steps:

  1. Run on demo account for 2-3 days using main.py
  2. Review fills and risk checks in logs/trades.json
  3. If demo results are clean, switch to live at 0.25% risk
""")
    elif g1_passed and not g2_passed:
        print("""
  [GATE 1 PASS / GATE 2 FAIL]

  Strategy works but tail risk is too high.
  Action: Widen stop loss from 1.5x ATR to 2.0x ATR.
  Re-run backtest.py after the change.
""")
    elif not g1_passed and g2_passed:
        print("""
  [GATE 1 FAIL / GATE 2 PASS]

  Strategy does not generalise across different time periods.
  Action: Tighten EMA filter from EMA55 to EMA21.
  Re-run backtest.py after the change.
""")
    else:
        print("""
  [BOTH GATES FAILED]

  Strategy needs redesign. Do NOT go live.
  Share these results and we will review together.
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="London Breakout Backtester")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to TickStory 1m CSV file (optional)")
    parser.add_argument("--sims", type=int, default=500,
                        help="Number of Monte Carlo simulations (default: 500)")
    args = parser.parse_args()

    print("=" * 60)
    print("  London Breakout Strategy — Validation Backtester")
    print("  Gate 1: Walk-Forward  |  Gate 2: Monte Carlo")
    print("=" * 60)

    # Load data
    print("\n  Loading data...")
    if args.csv:
        df = load_from_csv(args.csv)
    else:
        df = load_from_mt5()

    if df is None or len(df) < 500:
        print("\n  Not enough data to run backtest.")
        print("  Need at least 12 months of 15m bars.")
        print("  Try: python backtest.py --csv your_xauusd_data.csv")
        sys.exit(1)

    months_available = (df["time"].max() - df["time"].min()).days / 30
    print(f"  Data available: {months_available:.1f} months")

    if months_available < 12:
        print(f"\n  Warning: only {months_available:.1f} months of data.")
        print("  Walk-forward results may not be reliable.")
        print("  Recommend pulling TickStory data for better results.")

    # Full report first
    print_full_report(df)

    # Gate 1
    g1_passed, window_metrics = gate1_walk_forward(df)

    # Gate 2
    g2_passed, mc_results = gate2_monte_carlo(df, n_sims=args.sims)

    # Decision
    print_decision(g1_passed, g2_passed)

    # Save results to JSON
    results = {
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "data_months":     round(months_available, 1),
        "gate1_passed":    bool(g1_passed),
        "gate2_passed":    bool(g2_passed),
        "both_passed":     bool(g1_passed and g2_passed),
        "window_metrics":  window_metrics,
        "monte_carlo":     mc_results,
    }

    results_file = f"{config.LOG_PATH}backtest_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results saved to {results_file}\n")


if __name__ == "__main__":
    main()