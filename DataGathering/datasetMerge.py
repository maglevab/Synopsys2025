import pandas as pd

litDatasetAddOn = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/LinRegData/suppLitDatav2.csv")
raid = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/LinRegData/trainRandomv4.csv")
acadAddOn = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/LinRegData/abstractsScientificv2.csv")
newsAddOn = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv4.csv")



raidAcad = raid[raid["domain"] == "abstracts"]
raidAcadHuman = raidAcad[raidAcad["label"] == 0]
raidAcadHuman = raidAcadHuman[["label", "text"]]
raidAcadAI = raidAcad[raidAcad["label"] == 1]
raidAcadAI = raidAcadAI[["label", "text"]]
acadHumanAddOn = acadAddOn[acadAddOn["label"] == 0]
acadAIAddOn = acadAddOn[acadAddOn["label"] == 1]
acadHumanAddOn = acadHumanAddOn[:210]
acadAIAddOn = acadAIAddOn[:99]


raidLit = raid[raid["domain"].isin(["books", "poetry"])]
raidLitHuman = raidLit[raidLit["label"] == 0]
raidLitHuman = raidLitHuman[["label", "text"]]
raidLitAI = raidLit[raidLit["label"] == 1]
raidLitAI = raidLitAI[["label", "text"]]
raidLitAI = raidLitAI.sample(n = 215, random_state = 1)
litHumanAddOn = litDatasetAddOn[:211]

raidNews = raid[raid["domain"].isin(["news", "wiki", "reviews", "reddit"])]
raidNewsHuman = raidNews[raidNews["label"] == 0]
raidNewsHuman = raidNewsHuman[["label", "text"]]
raidNewsAI = raidNews[raidNews["label"] == 1]
raidNewsAI = raidNewsAI[["label", "text"]]
raidNewsAI = raidNewsAI.sample(n = 100, random_state = 1)

newsHumanAddOn = newsAddOn[newsAddOn["label"]==0]
newsAIAddOn = newsAddOn[newsAddOn["label"]==1]
newsHumanAddOn = newsHumanAddOn[:206]
newsAIAddOn = newsAIAddOn[:115]

masterDataset = pd.DataFrame({
    "text": [],
    "label": []
})
print(raidAcadHuman)
masterDataset = pd.concat([masterDataset, raidAcadHuman, raidAcadAI, raidLitHuman, raidLitAI, raidNewsHuman, raidNewsAI], ignore_index=True)
#add ons
masterDataset = pd.concat([masterDataset, acadHumanAddOn, acadAIAddOn, litHumanAddOn, newsHumanAddOn, newsAIAddOn])
print(masterDataset["label"].value_counts())
masterDataset.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/LinRegData/masterDatasetv3.csv")