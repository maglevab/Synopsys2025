import pandas as pd

with open("/Users/arahan/Downloads/archive (2)/castleofotranto.txt", "r") as file:
    text = file.read()

litDataset = { "text": [],
               "label": []}

litDataset = pd.DataFrame(litDataset)
print(len(text))
prevIndex = 0
counter = 0
for i in range(1000, len(text), 1000):
    tempText = text[prevIndex: i]
    tempText = tempText.replace("\n", " ")
    litDataset.loc[counter] = [tempText, 0]
    prevIndex = i
    counter += 1

print(litDataset.head())
litDataset.to_csv("/Users/arahan/Desktop/Synopsys 2025/Data/suppLitDatav2.csv")