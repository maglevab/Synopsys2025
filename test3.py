import pandas as pd
import joblib
from evaluate import load
import numpy as np
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification


def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilites = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilites, dim=-1).item()
        return predicted_class, probabilites.tolist()


testSmall = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/3000sampletestv2.csv")


testSmall = testSmall.sample(frac=1, ignore_index=True)
testSmall = testSmall.reset_index(drop=True)

testSmall = testSmall[["text", "label"]]

modelName = "/Users/arahan/Desktop/Synopsys2025/ExpertModels/AcademicModelv2"
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForSequenceClassification.from_pretrained(modelName)
model.eval()
correctai = 0
correcthuman = 0
ai = 0
human = 0
for i in range(len(testSmall)):
    text = testSmall["text"][i]
    label = testSmall["label"][i]

    pred, probs = predict_text(text)
    print(pred)
    print(probs)
    if label == 0:
        if pred == label:
            correcthuman += 1
        human += 1
    elif label == 1:
        if pred == label:
            correctai += 1
        ai += 1

print("ai stats")
print(correctai)
print(ai)
print(correctai/ai)
print("human stats")
print(correcthuman)
print(human)
print(correcthuman/human)
