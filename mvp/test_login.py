import MetaTrader5 as mt5

print("Init:", mt5.initialize())
print("Account:", mt5.account_info())
mt5.shutdown()