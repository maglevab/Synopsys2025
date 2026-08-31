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

with open(path + "The Fellowship Of The Ring.txt", "r", encoding="latin-1") as file1, open(path + "war of the worlds.txt", "r", encoding="latin-1") as file2, open(path+"book-war-and-peace (1).txt", "r", encoding = "latin-1") as file3, open(path+"autobioOfYogi.txt", "r", encoding="latin-1") as file4:
    lotr = file1.read()
    worlds = file2.read()
    peace = file3.read()
    yogi = file4.read()
litDataset = { "text": [],
               "label": []}
litDataset = pd.DataFrame(litDataset)

run(lotr, 0)
run(worlds, len(litDataset))
run(peace, len(litDataset))
run(yogi, len(litDataset))
print(litDataset)

litDataset.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDatasetHUMAN2.csv")