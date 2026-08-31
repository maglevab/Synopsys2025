import pandas as pd
import requests

raidMaster = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv2.csv")
raid = raidMaster.sample(n = 900, random_state=12)
raid.reset_index(drop=True, inplace=True)
#Weight Sets
sciWeights = [1.0, 1.0, 1.0]
acadWeights = [1.0, 1.0, 1.0]
litWeights = [1.0, 1.0, 1.0]

#Expert Prompts
classPrompt = "Remember examples of Academic Text such as essays or school assignments as “category 0 text”, Remember examples of Fiction such as novels or stories as “Category 1 text”, Remember examples of Scientific texts such as research papers or abstracts as “category 2 text”. using these categories, analyse the given text and output the best fit category, do not add reasoning, but only the string containing the word category and the number. Here is the text: "
#sciPrompt = "You are a model that specializes in detecting AI-generated scientific texts. This means scientific or research papers or abstract. I will give you the classification of the text and the text itself. I will give you a number and the text. If the number is 2, do your best to determine whether the text is ai-generated or not. If not, flip a coin and randomly decide whether the text is ai-generated or not. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text and number:  "
sciPrompt = "You are a model that specializes in detecting AI-generated scientific texts. This means scientific or research papers or abstract. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "
#acadPrompt = "You are a model that specializes in detecting AI-generated academic texts. This means school assignments or essays. I will give you the classification of the text and the text itself. I will give you a number and the text. If the number is 0, do your best to determine whether the text is ai-generated or not. If not, flip a coin and randomly decide whether the text is ai-generated or not. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text and number:  "
acadPrompt = "You are a model that specializes in detecting AI-generated academic texts. This means school assignments or essays. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "
#litPrompt = "You are a model that specializes in detecting AI-generated literature, both fiction and non-fiction. This means non-fiction books, but also novels and stories. I will give you the classification of the text and the text itself. I will give you a number and the text. If the number is 1, do your best to determine whether the text is ai-generated or not. If not, flip a coin and randomly decide whether the text is ai-generated or not. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai=generated text. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text and number: "
litPrompt = "You are a model that specializes in detecting AI-generated literature, both fiction and non-fiction. This means non-fiction books, but also novels and stories. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai=generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "

prompts = [sciPrompt, acadPrompt, litPrompt]
#Gets response from llama Mode
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

#selects which set of weights to use
def selectWeights(classNum):
    if classNum == 0:
        return acadWeights
    if classNum == 1:
        return litWeights
    return sciWeights

def updateWeights(classNum, weights):
    global acadWeights, litWeights, sciWeights
    if classNum == 0:
        acadWeights = weights
    elif classNum == 1:
        litWeights = weights
    else:
        sciWeights = weights
labels = raid["model"]
texts = raid["generation"]
weightTuner = 0.75

print(len(texts))
print(len(raid))

#returns list of responses from each model
def get_result(text, weights):
    responses = []
    whileCount = 0
    for i in range(len(weights)):
        answer = ""
        while answer.lower() != "class 0" and answer.lower() != "class 1" and answer.lower() != "class 9":
            if whileCount == 10:
                answer = "class 9"
                continue
            answer = llamaModel(prompts[i] + " " + str(classNum) + " " + text)
            print(answer)
            whileCount += 1
        answer = int(answer[6])
        responses.append(float(answer))
    return responses

weights = []
#running the weights
for i in range(len(raid)):
    print("weights")
    print(weights)
    classNum = llamaModel(classPrompt + texts[i])
    print(classNum)
    if classNum[9] != "0" and classNum[9] != "1" and classNum[9] != "2":
        print("skipped")
        continue

    print("didn't skip")
    classNum = int(classNum[9])
    weights = selectWeights(classNum)
    #iterate through all the models and get their response
    answers = get_result(texts[i], weights)
    print("answers")
    print(answers)
    for j in range(len(answers)):
        trueAnswer = 0
        if labels[i] == "human": trueAnswer = 0.0
        else: trueAnswer = 1.0
        if answers[j] == 9.0: continue
        if answers[j] != trueAnswer:
            weights[j] *= weightTuner

    updateWeights(classNum, weights)

print(acadWeights)
print(litWeights)
print(sciWeights)


