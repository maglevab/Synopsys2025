import pandas as pd
import numpy as np
import llamaModelCall

df = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/LinRegData/masterDatasetv2.csv")
df2 = df.sample(n = 615, random_state = 50, ignore_index=True)
df2 = df2.iloc[415:]
df2.reset_index(inplace=True)

weightsSci = np.array([0.54336507, 0.19682888, -0.05369915])
interceptSci = 0.34437896512163774

weightsSM = np.array([0.04437436, 0.25381472, 0.42000824])
interceptSM = 0.27897526299522657

weightsLit = np.array([0.14830789, 0.34562166, 0.34353576])
interceptLit = 0.20080768927300202

#test is np array of model responses
def run_prediction(weights, intercept, test):
    return intercept + np.dot(test, weights)

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
    print("classNum ", class_num)
    totalPredicted[class_num] += 1
    totalActual[int(answer)] += 1
    classes[class_num] += 1
    if class_num == 0:
        weights = weightsSci
        intercept = interceptSci
        sm, sci, lit = llamaModelCall.process_responses((llamaModelCall.smPrompt, llamaModelCall.sciPrompt, llamaModelCall.litPrompt), text)
        if sm == -1.0 or sci == -1.0 or lit == -1.0:
            print("skipped")
            continue
        testAnswers = np.array([sm, sci, lit])
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
        sm, sci, lit = llamaModelCall.process_responses((llamaModelCall.smPrompt, llamaModelCall.sciPrompt, llamaModelCall.litPrompt), text)
        if sm == -1.0 or sci == -1.0 or lit == -1.0:
            print("skipped")
            continue
        testAnswers = np.array([sm, sci, lit])
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
        sm, sci, lit = llamaModelCall.process_responses((llamaModelCall.smPrompt, llamaModelCall.sciPrompt, llamaModelCall.litPrompt), text)
        if sm == -1.0 or sci == -1.0 or lit == -1.0:
            print("skipped")
            continue
        testAnswers = np.array([sm, sci, lit])
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