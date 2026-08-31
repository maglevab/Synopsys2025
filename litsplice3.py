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
        litDataset.loc[i+addOn] = [files[i], 0]

path = "/Users/arahan/Downloads/"

with open(path + "tellTaleHeart.txt", "r", encoding="latin-1") as file1, open(path+"theOdyssey.txt", "r", encoding="latin-1") as file2:
    lotr = file1.read()
    odyssey = file2.read()

litDataset = { "text": [],
               "label": []}
litDataset = pd.DataFrame(litDataset)

run(lotr, 0)
run(odyssey, len(litDataset))
litDataset = litDataset.sample(n=300, ignore_index=True, random_state=5)
print(litDataset)
litDataset.to_csv("/Users/arahan/Desktop/Synopsys2025/testData/litDatasetHUMAN.csv")