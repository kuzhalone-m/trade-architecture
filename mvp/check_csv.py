import pandas as pd
df = pd.read_csv(r"C:\Users\marke\OneDrive\Documents\XAUUSD_mt5_ticks.csv", nrows=5)
print(df.columns.tolist())
print(df.head())