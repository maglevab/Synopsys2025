import pandas as pd

academic = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/abstractsScientificv2.csv")
print(len(academic["label"]))