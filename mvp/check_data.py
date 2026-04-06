import MetaTrader5 as mt5
import pandas as pd

mt5.initialize(login=51522, password="Test123@", server="AurumMarkets-Demo")

print("Last error:    ", mt5.last_error())
print("Account:       ", mt5.account_info())
print()

# Check if symbol exists and is visible
info = mt5.symbol_info("XAUUSD")
print("Symbol info:   ", info)
print()

# Try different bar counts and timeframes
for tf_name, tf in [("M15", mt5.TIMEFRAME_M15), ("M1", mt5.TIMEFRAME_M1), ("H1", mt5.TIMEFRAME_H1)]:
    rates = mt5.copy_rates_from_pos("XAUUSD", tf, 0, 100)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        print(f"{tf_name}: {len(rates)} bars | {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    else:
        print(f"{tf_name}: No data — error: {mt5.last_error()}")

mt5.shutdown()
