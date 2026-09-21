import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

raidMaster = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv2.csv")
raid = raidMaster[raidMaster["domain"] == "abstracts"]
raid.reset_index(drop=True, inplace=True)

classPrompt = "Remember examples of Academic Text such as essays or school assignments as “category 0 text”, Remember examples of Scientific texts such as research papers or abstracts as “Category 1 text”, Remember examples of Fiction such as novels or stories as “Category 2 text”. using these categories, analyse the given text and output the best fit category, do not add reasoning, but only the string containing the word category and the number. Here is the text: "
sciPrompt = "You are a model that specializes in detecting AI-generated scientific texts. This means scientific or research papers or abstract. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "
acadPrompt = "You are a model that specializes in detecting AI-generated academic texts. This means school assignments or essays. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "
litPrompt = "You are a model that specializes in detecting AI-generated literature, both fiction and non-fiction. This means non-fiction books, but also novels and stories. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai=generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "
url = "http://localhost:11434/api/chat"

def llamaModel(prompt):
    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']


acadResponsesA = np.array([])
acadResponsesS = np.array([])
acadResponsesL = np.array([])
sciResponsesA = np.array([])
sciResponsesS = np.array([])
sciResponsesL = np.array([])
litResponsesA = np.array([])
litResponsesS= np.array([])
litResponsesL = np.array([])
labelsA = np.array([])
labelsS = np.array([])
labelsL = np.array([])

#filling the datasets

for i in range(len(raid["generation"])):
    print(i)

    #Classify data
    shouldContinue = False

    whileCount = 0
    classNum = llamaModel(classPrompt + raid["generation"][i])
    classNum = classNum[9]
    while classNum != "0" and classNum != "1" and classNum != "2":
        print("redoing...")
        classNum = llamaModel(classPrompt + raid["generation"][i])
        classNum = classNum[9]
        whileCount += 1
        if whileCount == 10:
            shouldContinue = True
            classNum = "1"


    classNum = int(classNum)

    classNum = 0
    if shouldContinue:
        continue
    #label data
    print(classNum)
    trueAnswer = 0
    if raid["model"][i] == "human":
        trueAnswer = 0.0
    else:
        trueAnswer = 1.0
    if classNum == 0:
        print("appending labelsA")
        labelsA = np.append(labelsA, trueAnswer)
    elif classNum == 1:
        labelsS = np.append(labelsS, trueAnswer)
    elif classNum == 2:
        labelsL = np.append(labelsL, trueAnswer)
    #if its academic
    if int(classNum) == 0:
        print("calling models academic")
        answer = llamaModel(acadPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(acadPrompt + raid['generation'][i])
            print(answer)

        answer = float(answer[6])
        acadResponsesA = np.append(acadResponsesA, answer)
        answer = llamaModel(sciPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(sciPrompt + raid['generation'][i])
            print(answer)

        answer = float(answer[6])
        sciResponsesA = np.append(sciResponsesA, answer)
        answer = llamaModel(litPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(litPrompt + raid['generation'][i])
            print(answer)

        answer = float(answer[6])
        litResponsesA = np.append(litResponsesA, answer)
    #if its scientific
    elif int(classNum) == 1:
        print("calling models scientific")
        answer = llamaModel(acadPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(acadPrompt + raid['generation'][i])
            print(answer)

        answer = float(answer[6])
        acadResponsesS = np.append(acadResponsesS, answer)
        answer = llamaModel(sciPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(sciPrompt + raid['generation'][i])
            print(answer)

        answer = float(answer[6])
        sciResponsesS = np.append(sciResponsesS, answer)
        answer = llamaModel(litPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(litPrompt + raid['generation'][i])
            print(answer)

        answer = float(answer[6])
        litResponsesS = np.append(litResponsesS, answer)
    #if its literature
    elif int(classNum) == 2:
        print("calling models literature")
        answer = llamaModel(acadPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(acadPrompt + raid['generation'][i])
            print(answer)
        answer = float(answer[6])
        acadResponsesL = np.append(acadResponsesL, answer)
        answer = llamaModel(sciPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(sciPrompt + raid['generation'][i])
            print(answer)
        answer = float(answer[6])
        sciResponsesL = np.append(sciResponsesL, answer)
        answer = llamaModel(litPrompt + raid["generation"][i])
        answerCount = 0
        while answer != "class 0" and answer != "class 1":
            print("redoing answer... " + str(answerCount))
            answerCount += 1
            answer = llamaModel(acadPrompt + raid['generation'][i])
            print(answer)
        answer = float(answer[6])
        litResponsesL = np.append(litResponsesL, answer)
    else:
        continue


print(len(acadResponsesA))
print(len(sciResponsesA))
print(len(litResponsesA))
'''
print(len(acadResponsesS))
print(len(sciResponsesS))
print(len(litResponsesS))
print(len(acadResponsesL))
print(len(sciResponsesL))
print(len(litResponsesL))
'''
print(len(labelsA))
#training academic linear reg
Xa = np.column_stack((acadResponsesA, sciResponsesA, litResponsesA))
#Xa = np.hstack((np.ones((Xa.shape[0], 1)), Xa))

#Xat = Xa.T
#weightsA = np.linalg.inv(Xat @ Xa) @ Xat @ labelsA

print(Xa)
print(labelsA)
modelA = LinearRegression()
modelA.fit(Xa, labelsA)
weightsA = modelA.coef_
interceptA = modelA.intercept_
print(interceptA)
print(weightsA)
print()

'''
#training scientific linear reg
Xs = np.column_stack((acadResponsesS, sciResponsesS, litResponsesS))
Xs = np.hstack((np.ones((Xs.shape[0], 1)), Xs))

Xst = Xs.T
weightsS = np.linalg.inv(Xst @ Xs) @ Xst @ labelsS

print(weightsS)
interceptS = weightsS[0]
coefficientsS = weightsS[1:]
print()

#training literature linear reg
Xl = np.column_stack((acadResponsesL, sciResponsesL, litResponsesL))
Xl = np.hstack((np.ones((Xl.shape[0], 1)), Xl))

Xlt = Xl.T
weightsL = np.linalg.inv(Xlt @ Xl) @ Xlt @ labelsL

print(weightsL)
interceptL = weightsL[0]
coefficientsL = weightsL[1:]
print()

masterWeights = [weightsA, weightsS, weightsL]
'''