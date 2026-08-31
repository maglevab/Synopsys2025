import numpy as np
import pandas as pd
import llamaModelCall
import torch
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
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

classDF = {
    "class": []
}

classDF = pd.DataFrame(classDF)

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

tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
academicExpert = DebertaV2ForSequenceClassification.from_pretrained("/Users/arahan/Desktop/Synopsys2025/ExpertModels/AcademicModelv2")
newsExpert = DebertaV2ForSequenceClassification.from_pretrained("/Users/arahan/Desktop/Synopsys2025/ExpertModels/NewsModelv1")
litExpert = DebertaV2ForSequenceClassification.from_pretrained("/Users/arahan/Desktop/Synopsys2025/ExpertModels/LitModelv2")
academicExpert.eval()
newsExpert.eval()
litExpert.eval()
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

def predict_text(model, text):
    global tokenizer
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilites = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilites, dim=-1).item()
        return predicted_class

for i, text in enumerate(test["text"]):
    print(i)
    print(text)

    # Classify data
    class_num = llamaModelCall.classify_data(test["text"][i])
    classDF.loc[i] = [class_num]
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
    
    print("calling models")
    sm = predict_text(newsExpert, text)
    sci = predict_text(academicExpert, text)
    lit = predict_text(litExpert, text)
    # Handle response
    #based on classification
    if class_num == 0:
        smResponsesS = np.append(smResponsesS, sm)
        sciResponsesS = np.append(sciResponsesS, sci)
        litResponsesS = np.append(litResponsesS, lit)
        print("appended responses scientific")
    elif class_num == 1:
        smResponsesM = np.append(smResponsesM, sm)
        sciResponsesM = np.append(sciResponsesM, sci)
        litResponsesM = np.append(litResponsesM, lit)
        print("appended responses media")
    elif class_num == 2:
        smResponsesL = np.append(smResponsesL, sm)
        sciResponsesL = np.append(sciResponsesL, sci)
        litResponsesL = np.append(litResponsesL, lit)
        print("appended responses literature")

print("done! saving arrays now")
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/smResponsesS", smResponsesS)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/sciResponsesS", sciResponsesS)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/litResponsesS", litResponsesS)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/smResponsesM", smResponsesM)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/sciResponsesM", sciResponsesM)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/litResponsesM", litResponsesM)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/smResponsesL", smResponsesL)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/sciResponsesL", sciResponsesL)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/litResponsesL", litResponsesL)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/labelsS", labelsS)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/labelsM", labelsM)
np.save("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/labelsL", labelsL)
classDF.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/classDF3000.csv")

print("saved arrays")