import numpy as np
import pandas as pd
import joblib
import llamaModelCallv2
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

#takes piece of text and expert model and outputs a response (either 1 or 0)
def predict_text(model, text):
    global tokenizer
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class#, probabilities

#load in test data
test_data = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/3000sampletestv2.csv")
test_data = test_data.sample(n=900, ignore_index=True)
test_data = test_data.reset_index(drop=True)
#initialize compilers and expert model
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

classes = [0, 0, 0]

correctAClasses = [0, 0, 0]
correctMClasses = [0, 0, 0]
correctLClasses = [0, 0, 0]

correctA = 0
correctM = 0
correctL = 0

truePA, trueNA, falsePA, falseNA = 0, 0, 0, 0
truePM, trueNM, falsePM, falseNM = 0, 0, 0, 0
truePL, trueNL, falsePL, falseNL = 0, 0, 0, 0

for i in range(len(test_data)):
    print(i)
    text = test_data["text"][i]
    label = test_data["label"][i]
    classNum = llamaModelCallv2.classify_data(text)
    classes[classNum] += 1
    acadAnswer = predict_text(academicExpert, text)
    mediaAnswer = predict_text(mediaExpert, text)
    litAnswer = predict_text(literatureExpert, text)
    print([acadAnswer, mediaAnswer, litAnswer])
    if acadAnswer == label:
        correctA += 1
        if label == 1: truePA += 1
        else: trueNA += 1
    else:
        if acadAnswer == 1: falsePA += 1
        else: falseNA += 1

    if mediaAnswer == label:
        correctM += 1
        if label == 1: truePM += 1
        else: trueNM += 1
    else:
        if mediaAnswer == 1: falsePM += 1
        else: falseNM += 1

    if litAnswer == label:
        correctL += 1
        if label == 1: truePL += 1
        else: trueNL += 1
    else:
        if litAnswer == 1: falsePL += 1
        else: falseNL += 1

    if classNum == 0:
        if acadAnswer == label: correctAClasses[0] += 1
        if mediaAnswer == label: correctMClasses[0] += 1
        if litAnswer == label: correctLClasses[0] += 1
    elif classNum == 1:
        if acadAnswer == label: correctAClasses[1] += 1
        if mediaAnswer == label: correctMClasses[1] += 1
        if litAnswer == label: correctLClasses[1] += 1
    elif classNum == 2:
        if acadAnswer == label: correctAClasses[2] += 1
        if mediaAnswer == label: correctMClasses[2] += 1
        if litAnswer == label: correctLClasses[2] += 1

def precision_recall_f1(tp, fp, fn):
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = (2*precision*recall)/(precision+recall)
    return precision*100, recall*100, f1*100

precisionA, recallA, f1A = precision_recall_f1(truePA, falsePA, falseNA)
precisionM, recallM, f1M = precision_recall_f1(truePM, falsePM, falseNM)
precisionL, recallL, f1L = precision_recall_f1(truePL, falsePL, falseNL)

print("Academic")
print("Accuracy: {}. Precision: {}. Recall: {}. F1 Score: {}".format(correctA/len(test_data), precisionA, recallA, f1A))

print("Media")
print("Accuracy: {}. Precision: {}. Recall: {}. F1 Score: {}".format(correctM/len(test_data), precisionM, recallM, f1M))

print("Literature")
print("Accuracy: {}. Precision: {}. Recall: {}. F1 Score: {}".format(correctL/len(test_data), precisionL, recallL, f1L))


print("Accuracy scores per category")
print("Academic Expert in Academic Category: {}, in Media Category: {}, in Literature Category: {}/".format((correctAClasses[0]/classes[0])*100, (correctAClasses[1]/classes[1])*100, (correctAClasses[2]/classes[2])*100))
print("Media Expert in Academic Category: {}, in Media Category: {}, in Literature Category: {}/".format((correctMClasses[0]/classes[0])*100, (correctMClasses[1]/classes[1])*100, (correctMClasses[2]/classes[2])*100))
print("Literature Expert in Academic Category: {}, in Media Category: {}, in Literature Category: {}/".format((correctLClasses[0]/classes[0])*100, (correctLClasses[1]/classes[1])*100, (correctLClasses[2]/classes[2])*100))
