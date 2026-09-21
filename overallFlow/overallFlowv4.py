import numpy as np
import pandas as pd
import joblib
import llamaModelCallv2
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

#load in test dataset
academic = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/abstractsScientificv2.csv")
academicSample, _ = train_test_split(academic, train_size=300, stratify=academic["label"], random_state=15)
news = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv5.csv")
newsSample, _ = train_test_split(news, train_size=300, stratify=news['label'], random_state=15)
litH = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDatasetHUMAN.csv")
litHSample, _ = train_test_split(litH, train_size=150, random_state=15)
litAI = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDataAI.csv")
litAISample, _ = train_test_split(litAI, train_size=150, random_state=15)

test = pd.concat([academicSample, newsSample, litHSample, litAISample], ignore_index=True)
print(test.head())
print(len(test))
testLength=len(test)
classes = [0, 0, 0]
#load in compilers
AcademicCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModelsv2/AcademicCompiler.pkl")
MediaCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModelsv2/MediaCompiler.pkl")
LiteratureCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModelsv2/LiteratureCompiler.pkl")

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
academicExpert = DebertaV2ForSequenceClassification.from_pretrained('/Users/arahan/Desktop/Synopsys2025/ExpertModels/AcademicModelv2')
mediaExpert = DebertaV2ForSequenceClassification.from_pretrained('/Users/arahan/Desktop/Synopsys2025/ExpertModels/newsModelv2')
literatureExpert = DebertaV2ForSequenceClassification.from_pretrained('/Users/arahan/Desktop/Synopsys2025/ExpertModels/LitModelv2')
academicExpert.eval()
mediaExpert.eval()
literatureExpert.eval()

def predict_text(model, text):
    global tokenizer
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class#, probabilities

acad = 0
news = 0
lit = 0
def runFlow(text):

    #classify text
    classNum = llamaModelCallv2.classify_data(text)
    classes[classNum] += 1
    print("classified as " + str(classNum))
    #query expert models
    academicAnswer = predict_text(academicExpert, text)
    mediaAnswer = predict_text(mediaExpert, text)
    literatureAnswer = predict_text(literatureExpert, text)
    #use synthesizer
    answers = np.array([[academicAnswer, mediaAnswer, literatureAnswer]])
    print(answers)
    if classNum == 0:
        finalAnswer = max(AcademicCompiler.predict(answers))
    elif classNum == 1:
        finalAnswer = max(MediaCompiler.predict(answers))
    elif classNum == 2:
        finalAnswer = max(LiteratureCompiler.predict(answers))
    else:
        print("Not able to classify text, please check your input.")
        finalAnswer = -1

    return finalAnswer, classNum

correctA = 0
correctM = 0
correctL = 0
totalPredicted = [0,0,0]
totalActual = [0,0,0]
totalCorrect = [0,0,0]

for i in range(testLength):
    print(i)
    text = test["text"][i]
    finalAnswer, classNum = runFlow(text)
    if finalAnswer == -1:
        continue
    totalPredicted[classNum] += 1
    totalActual[test["label"][i]] += 1
    if finalAnswer == test["label"][i]:
        totalCorrect[classNum] += 1
        if classNum == 0:
            correctA += 1
        elif classNum == 1:
            correctM += 1
        elif classNum == 2:
            correctL += 1
    print(finalAnswer)

precision = [totalCorrect[i]/totalPredicted[i] if totalPredicted[i]>0 else 0 for i in range(3)]
recall = [totalCorrect[i]/totalActual[i] if totalActual[i] > 0 else 0 for i in range(3)]
overall_precision = sum(totalCorrect)/sum(totalPredicted) if sum(totalPredicted) > 0 else 0
overall_recall = sum(totalCorrect) / sum(totalActual) if sum(totalActual) > 0 else 0

print("academic")
print(correctA)
print(classes[0])
print(correctA/classes[0])
print(precision[0])
print(recall[0])

print("media")
print(correctM)
print(classes[1])
print(correctM/classes[1])
print(precision[1])
print(recall[1])

print("literature")
print(correctL)
print(classes[2])
print(correctL/classes[2])
print(precision[2])
print(recall[2])

print()
print(overall_precision)
print(overall_recall)