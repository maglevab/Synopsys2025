import pandas as pd
import numpy as np
import expertModelCall
import llamaModelCall
import joblib
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

df = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/LinRegData/masterDatasetv2.csv")
df2 = df.sample(n = 615, random_state = 50, ignore_index=True)
df2 = df2.iloc[415:]
df2.reset_index(inplace=True)

weightsSM = np.array([0.85464837, 0.29281397, 0.0])
interceptSM = 0.06619975055166416

weightsSci = np.array([0.21398374, 0.62601626, 0.0])
interceptSci = 0.16000000000000092

weightsLit = np.array([-0.40022057,  0.58805433,  0.50801022])
interceptLit = 0.812166241003018

#test is np array of model responses
def run_prediction(weights, intercept, test):
    return intercept + np.dot(test, weights)

def predict_text(model, text):
    global tokenizer
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class#, probabilities

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
correctA = 0
correctM = 0
correctL = 0
totalPredicted = [0,0,0]
totalActual = [0,0,0]
totalCorrect = [0,0,0]

for i in range(len(df2)):
    print(i)
    text = df2["text"][i]
    answer = df2["label"][i]
    print(answer)
    class_num = llamaModelCall.classify_data(text)
    if class_num == -1:
        continue
    totalPredicted[class_num] += 1
    totalActual[int(answer)] += 1
    classes[class_num] += 1
    print("classNum ", class_num)
    sm = predict_text(mediaExpert, text)
    sci = predict_text(academicExpert, text)
    lit = predict_text(literatureExpert, text)
    testAnswers = np.array([sm, sci, lit])
    if class_num == 0:
        weights = weightsSci
        intercept = interceptSci
        print(testAnswers)
        if run_prediction(weights, intercept, testAnswers) >= 0.5:
            testAnswer = 1.0
        else:
            testAnswer = 0.0
        if testAnswer == answer:
            correctA += 1
            totalCorrect[class_num] += 1

    elif class_num == 1:
        weights = weightsSM
        intercept = interceptSM
        print(testAnswers)
        if run_prediction(weights, intercept, testAnswers) >= 0.5:
            testAnswer = 1.0
        else:
            testAnswer = 0.0
        if testAnswer == answer:
            correctM += 1
            totalCorrect[class_num] += 1

    elif class_num == 2:
        weights = weightsLit
        intercept = interceptLit
        print(testAnswers)
        if run_prediction(weights, intercept, testAnswers) >= 0.5:
            testAnswer = 1.0
        else:
            testAnswer = 0.0
        if testAnswer == answer:
            correctL += 1
            totalCorrect[class_num] += 1


precision = [totalCorrect[i]/totalPredicted[i] if totalPredicted[i]>0 else 0 for i in range(3)]
recall = [totalCorrect[i]/totalActual[i] if totalActual[i] > 0 else 0 for i in range(3)]
overall_accuracy = sum([correctA, correctM, correctL])/sum(classes)
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

print("overall")
print(overall_accuracy)
print(overall_precision)
print(overall_recall)