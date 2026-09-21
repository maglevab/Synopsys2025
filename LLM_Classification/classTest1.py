import pandas as pd
import llamaModelCallv2
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
acadDf = pd.read_csv('/Users/arahan/Desktop/Synopsys2025/Data/abstractsScientificv2.csv')
newsDf = pd.read_csv('/Users/arahan/Desktop/Synopsys2025/Data/newsv5.csv')
litDfAI = pd.read_csv('/Users/arahan/Desktop/Synopsys2025/Data/litDataAI.csv')
litDfH = pd.read_csv('/Users/arahan/Desktop/Synopsys2025/Data/litDatasetHUMAN.csv')

testAcad = acadDf.sample(n=100, ignore_index=True)
testNews = newsDf.sample(n=100, ignore_index=True)
testLitAI = litDfAI.sample(n=50, ignore_index=True)
testLitHuman = litDfH.sample(n=50, ignore_index=True)
testLit = pd.concat([testLitAI, testLitHuman], ignore_index=True)
correctA = 0
correctN = 0
correctL = 0

def predict_text(model, text):
    global tokenizer
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class#, probabilities

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
classifyModel = DebertaV2ForSequenceClassification.from_pretrained('/Users/arahan/Desktop/Synopsys2025/classModelv2')
classifyModel.eval()
classNum = 0
for i in range(len(testAcad)):
    text = testAcad["text"][i]
    classNum == predict_text(classifyModel, text)
    print(i, classNum)
    if classNum == 0:
        correctA += 1

for i in range(len(testNews)):
    text = testNews["text"][i]
    classNum == predict_text(classifyModel, text)
    print(i, classNum)
    if classNum == 1:
        correctN += 1

for i in range(len(testLit)):
    text = testLit["text"][i]
    classNum == predict_text(classifyModel, text)
    print(i, classNum)
    if classNum == 2:
        correctL += 1

print(correctA)
print(correctA/50)
print()
print(correctN)
print(correctN/50)
print()
print(correctL)
print(correctL/50)
