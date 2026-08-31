import pandas as pd

df = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv3.csv")
for i in range(len(df)):
    if df["label"][i] == "human":
        df["label"][i] = 0
    else:
        df["label"][i] = 1

df.to_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv4.csv")