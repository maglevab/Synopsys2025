import pandas as pd

path = "/Users/arahan/Desktop/Synopsys2025/testData/"

academicAI = pd.read_csv(path+"academicAI.csv")
academicHuman = pd.read_csv(path+"academicHuman.csv")
litAI = pd.read_csv(path+"litDataAI.csv")
litHuman = pd.read_csv(path+"litDatasetHUMAN.csv")
newsAI = pd.read_csv(path+"news_articles_full.csv")
newsHuman = pd.read_csv(path+"newsHuman.csv")

academicAI = academicAI[["text", "label"]]
academicHuman = academicHuman[["text", "label"]]
litAI = litAI[["text", "label"]]
litHuman = litHuman[["text", "label"]]
newsHuman = newsHuman.rename(columns={'Text Chunk': 'text'})
newsHuman["label"] = [0]*len(newsHuman)

test_data = pd.concat([academicAI, academicHuman, litAI, litHuman, newsHuman, newsAI], ignore_index=True)

test_data = test_data.sample(frac=1, ignore_index=True, random_state=10)
test_data.to_csv(path+"combinedTestData.csv")