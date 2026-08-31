import pandas as pd
from sklearn.model_selection import train_test_split

academic = pd.read_csv("/Users/arahan/Downloads/data_set.csv")
academic = academic[["abstract", "is_ai_generated"]]
academic = academic.rename(columns={'abstract': 'text', 'is_ai_generated': 'label'})
academic = academic.sample(frac=1.0, ignore_index=True)
academicHuman = pd.DataFrame({'text': [], 'label': []})
academicAI = pd.DataFrame({'text': [], 'label': []})

countH = 0
countAI = 0
for i in range(len(academic)):
    if countH == 300 and countAI == 300:
        break
    if academic["label"][i] == 0 and countH != 300:
        academicHuman.loc[i] = [academic["text"][i], academic["label"][i]]
        countH += 1
    elif academic["label"][i] == 1 and countAI != 300:
        academicAI.loc[i] = [academic["text"][i], academic["label"][i]]
        countAI += 1

academicHuman = academicHuman.reset_index(drop=True)
academicAI = academicAI.reset_index(drop=True)

academicHuman.to_csv('/Users/arahan/Desktop/Synopsys2025/testData/academicHuman.csv')
academicAI.to_csv('/Users/arahan/Desktop/Synopsys2025/testData/academicAI.csv')