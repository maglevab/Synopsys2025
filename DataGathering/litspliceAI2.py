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
with open("/Users/arahan/Downloads/ai-generated.txt") as file1, open(path + "Eternal_Star_Chronicles.txt", "r", encoding="latin-1") as file2:
    harryPotter = file1.read()
    book = file2.read()

litDataset = { "text": [],
               "label": []}
litDataset = pd.DataFrame(litDataset)

run(harryPotter, 0)
run(book, len(litDataset))
print(litDataset)

litDataset.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDataAIv2.csv")