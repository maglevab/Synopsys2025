import pandas as pd

raidMaster = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv2.csv")
raid = raidMaster.sample(n = 900, random_state=12)
raid.reset_index(drop=True, inplace=True)

texts = raid['generation']
print(texts.head())

for i in range(len(texts)):
    print(texts[i])