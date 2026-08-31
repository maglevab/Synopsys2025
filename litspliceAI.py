import pandas as pd

def run(file, addOn):
    global litDataset
    files = []
    label = []
    prevIndex = 0
    counter = 0
    for i in range(1000, len(file), 1000):
        tempFile = file[prevIndex: i]
        tempFile = tempFile.replace("\n", " ")
        files.append(tempFile)
        prevIndex = i
        counter += 1

    for i in range(len(files)):
        litDataset.loc[i+addOn] = [files[i], 1]

path = "/Users/arahan/Downloads/"
with open("/Users/arahan/Downloads/AIGENERATEDLIT.txt") as file1:
    harryPotter = file1.read()

litDataset = { "text": [],
               "label": []}
litDataset = pd.DataFrame(litDataset)

run(harryPotter, 0)

print(litDataset)

litDataset.to_csv("/Users/arahan/Desktop/Synopsys 2025/Data/litDataAI.csv")