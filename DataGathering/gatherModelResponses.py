import numpy as np
import pandas as pd
import llamaModelCall
import expertModelCallv2
from sklearn.model_selection import train_test_split

academic = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/abstractsScientificv2.csv")
academicSample, _ = train_test_split(academic, train_size=1000, stratify=academic["label"], random_state=5)
news = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv5.csv")
newsSample, _ = train_test_split(news, train_size=1000, stratify=news['label'], random_state=5)
litH = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDatasetHUMAN.csv")
litHSample, _ = train_test_split(litH, train_size=500, random_state=5)
litAI = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDataAI.csv")
litAISample, _ = train_test_split(litAI, train_size=500, random_state=5)

test = pd.concat([academicSample, newsSample, litHSample, litAISample], ignore_index=True)
print(len(test))
test.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/3000sampletestv2.csv")
print(test.head())
smResponsesM = np.array([])
smResponsesS = np.array([])
smResponsesL = np.array([])
sciResponsesM = np.array([])
sciResponsesS = np.array([])
sciResponsesL = np.array([])
litResponsesM = np.array([])
litResponsesS= np.array([])
litResponsesL = np.array([])
labelsM = np.array([])
labelsS = np.array([])
labelsL = np.array([])

def append_label(class_num, true_answer):
    """Append labels based on classification."""
    if class_num == 0:
        print("Appending to labelsS")
        return np.append(labelsS, true_answer)
    elif class_num == 1:
        print("Appending to labelsM")
        return np.append(labelsM, true_answer)
    elif class_num == 2:
        print("Appending to labelsL")
        return np.append(labelsL, true_answer)

def process_responses(model_names, text):
    """Process responses for a given set of prompts."""
    sm, sci, lit = model_names
    sm_response = expertModelCallv2.predict(text, sm)
    sci_response = expertModelCallv2.predict(text, sci)
    lit_response = expertModelCallv2.predict(text, lit)
    return sci_response, sm_response, lit_response


for i, text in enumerate(test["text"]):
    print(i)
    print(text)

    # Classify data
    class_num = llamaModelCall.classify_data(test["text"][i])
    print(class_num)
    if class_num == -1:
        print("skipped")
        continue
    if class_num == 0:
        labelsS = append_label(class_num, test["label"][i])
    elif class_num == 1:
        labelsM = append_label(class_num, test["label"][i])
    elif class_num == 2:
        labelsL = append_label(class_num, test["label"][i])

    # Handle response
    #based on classification
    print("calling models")
    sm, sci, lit = process_responses(("news_expert", "acad_expert", "lit_expert"), text)
    if class_num == 0:
        smResponsesS = np.append(smResponsesS, sm)
        sciResponsesS = np.append(sciResponsesS, sci)
        litResponsesS = np.append(litResponsesS, lit)
        print("appended responses scientific")
    elif class_num == 1:
        sm, sci, lit = process_responses(("news_expert", "acad_expert", "lit_expert"), text)
        smResponsesM = np.append(smResponsesM, sm)
        sciResponsesM = np.append(sciResponsesM, sci)
        litResponsesM = np.append(litResponsesM, lit)
        print("appended responses media")
    elif class_num == 2:
        sm, sci, lit = process_responses(("news_expert", "acad_expert", "lit_expert"), text)
        smResponsesL = np.append(smResponsesL, sm)
        sciResponsesL = np.append(sciResponsesL, sci)
        litResponsesL = np.append(litResponsesL, lit)
        print("appended responses literature")

print("done! saving arrays now")
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/smResponsesS", smResponsesS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/sciResponsesS", sciResponsesS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/litResponsesS", litResponsesS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/smResponsesM", smResponsesM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/sciResponsesM", sciResponsesM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/litResponsesM", litResponsesM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/smResponsesL", smResponsesL)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/sciResponsesL", sciResponsesL)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/litResponsesL", litResponsesL)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/labelsS", labelsS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/labelsM", labelsM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArraysLarge/labelsL", labelsL)

print("saved arrays")