import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

import llamaModelCallv2
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

#takes a piece of text and model responses, classifies it, queries experts, compiles model, and returns final answer
def runFlow(text, answers):

    global classes
    #classify text
    classNum = llamaModelCallv2.classify_data(text)
    classes[classNum] += 1
    print("classified as " + str(classNum))

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

#variables
classes = [0, 0, 0]
correctA = 0
correctM = 0
correctL = 0
totalPredicted = [0,0,0]
totalActual = [0,0,0]
totalCorrect = [0,0,0]

for i in range(len(test_data)):
    print(i)
    text = test_data["text"][i]
    label = test_data["label"][i]

    acadAnswer = predict_text(academicExpert, text)
    mediaAnswer = predict_text(mediaExpert, text)
    litAnswer = predict_text(literatureExpert, text)
    answers = np.array([[acadAnswer, mediaAnswer, litAnswer]])

    finalAnswer, classNum = runFlow(text, answers)

    if finalAnswer == -1:
        continue
    totalPredicted[classNum] += 1
    totalActual[label] += 1
    if finalAnswer == label:
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
overall_accuracy = sum([correctA, correctM, correctL])/(len(test_data))

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

print("overall")
print()
print(overall_precision)
print(overall_recall)