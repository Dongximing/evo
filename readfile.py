import pandas as pd



df = pd.read_pickle("/Users/ximing/Desktop/alpaca_v2.pickle")
print(len(df["data"]["alpaca_v2"]["correctness"]))