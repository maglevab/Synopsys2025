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

with open(path + "HarryPotterOrderofPh.txt", "r") as file1, open(path + "dune.txt", "r", encoding="latin-1") as file2, open(path+"steveJobs.txt", "r", encoding = "latin-1") as file3, open(path+"cagedBird.txt", "r", encoding="latin-1") as file4, open(path+'percyJackson.txt', 'r', encoding='latin-1') as file5, open(path+"greatGastby.txt", "r", encoding="latin-1") as file6, open(path+"archive (2)/castleofotranto.txt", "r", encoding="latin-1") as file7, open(path+"grapesOfWrath.txt", "r", encoding="latin-1") as file8:
    harryPotter = file1.read()
    dune = file2.read()
    steveJobs = file3.read()
    cagedBird = file4.read()
    percyJackson = file5.read()
    gatsby = file6.read()
    otranto = file7.read()
    grapes = file8.read()

litDataset = { "text": [],
               "label": []}
litDataset = pd.DataFrame(litDataset)

run(harryPotter, 0)
run(dune, len(litDataset))
run(steveJobs, len(litDataset))
run(cagedBird, len(litDataset))
run(percyJackson, len(litDataset))
run(gatsby, len(litDataset))
run(otranto, len(litDataset))
run(grapes, len(litDataset))
print(litDataset)

litDataset.to_csv("/Users/arahan/Desktop/Synopsys 2025/Data/litDatasetHUMAN.csv")