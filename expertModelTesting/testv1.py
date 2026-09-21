import pandas as pd
import llamaModelCall
import numpy as np

test_data = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/testData/combinedTestData.csv")

classes = [0, 0, 0]
correctA = 0
correctM = 0
correctL = 0
totalPredicted = [0,0,0]
totalActual = [0,0,0]
totalCorrect = [0,0,0]

truePA, trueNA, falsePA, falseNA = 0, 0, 0, 0
truePM, trueNM, falsePM, falseNM = 0, 0, 0, 0
truePL, trueNL, falsePL, falseNL = 0, 0, 0, 0

weightsSci = np.array([0.54336507, 0.19682888, -0.05369915])
interceptSci = 0.34437896512163774

weightsSM = np.array([0.04437436, 0.25381472, 0.42000824])
interceptSM = 0.27897526299522657

weightsLit = np.array([0.14830789, 0.34562166, 0.34353576])
interceptLit = 0.20080768927300202
correct = 0
finalAnswer = 0
#test is np array of model responses
def run_prediction(weights, intercept, test):
    return intercept + np.dot(test, weights)

for i in range(len(test_data)):
    text = test_data["text"][i]
    answer = test_data["label"][i]

    class_num = llamaModelCall.classify_data(text)
    sci, sm, lit = llamaModelCall.process_responses([llamaModelCall.smPrompt, llamaModelCall.sciPrompt, llamaModelCall.litPrompt], text)

    if sm == -1.0 or sci == -1.0 or lit == -1.0:
        print("skipped")
        continue

    if class_num == 0:
        weights = weightsSci
        intercept = interceptSci
        testAnswers = np.array([sm, sci, lit])
        print(testAnswers)
        finalAnswer =  run_prediction(weights, intercept, testAnswers)

    elif class_num == 1:
        weights = weightsSM
        intercept = interceptSM
        testAnswers = np.array([sm, sci, lit])
        print(testAnswers)
        finalAnswer =  run_prediction(weights, intercept, testAnswers)

    elif class_num == 2:
        weights = weightsLit
        intercept = interceptLit
        testAnswers = np.array([sm, sci, lit])
        print(testAnswers)
        finalAnswer =  run_prediction(weights, intercept, testAnswers)

    if finalAnswer == answer:
        correct += 1

    if sci == answer:
        correctA += 1
        if answer == 1: truePA += 1
        else: falseNA += 1
    else:
        if sci == 1: falsePA += 1
        else: falseNA += 1

    if sm == answer:
        correctM += 1
        if answer == 1: truePM += 1
        else: falseNM += 1
    else:
        if sm == 1: falsePM += 1
        else: falseNM += 1

    if lit == answer:
        correctL += 1
        if answer == 1: truePL += 1
        else: falseNL += 1
    else:
        if lit == 1: falsePL += 1
        else: falseNL += 1

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
