import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression


raidMaster = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv2.csv")
raid = raidMaster[raidMaster["domain"].isin(["abstracts", "books"])]
print(raid.head())
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

def classify_data(prompt, generation):
    """Classify the data using the model and handle retries."""
    while_count = 0
    while True:
        class_num = llamaModel(prompt + generation)
        class_num = class_num[9]
        if class_num in {"0", "1", "2"}:
            return int(class_num)
        print("Redoing classification...")
        while_count += 1
        if while_count == 10:
            print("Max retries reached. Defaulting to class 1.")
            return -1  # Default class


def append_label(class_num, true_answer):
    """Append labels based on classification."""
    if class_num == 0:
        print("Appending to labelsA")
        return np.append(labelsA, true_answer)
    elif class_num == 1:
        print("Appending to labelsS")
        return np.append(labelsS, true_answer)
    elif class_num == 2:
        print("Appending to labelsL")
        return np.append(labelsL, true_answer)


def call_model(prompt, generation):
    """Call the model and handle retries for responses."""
    answer_count = 0
    while True:
        answer = llamaModel(prompt + generation)
        if answer in {"class 0", "class 1"}:
            return float(answer[6])
        print(f"Redoing answer... {answer_count}")
        answer_count += 1
        if answer_count == 5:
            return -1.0


def process_responses(prompts, generation):
    """Process responses for a given set of prompts."""
    acad, sci, lit = prompts
    acad_response = call_model(acad, generation)
    sci_response = call_model(sci, generation)
    lit_response = call_model(lit, generation)
    return acad_response, sci_response, lit_response


for i, generation in enumerate(raid["generation"]):
    print(i)

    # Classify data
    class_num = classify_data(classPrompt, generation)
    if class_num == -1:
        continue
    if class_num == 0:
        labelsA = append_label(class_num, 0.0 if raid["model"][i] == "human" else 1.0)
    elif class_num == 1:
        labelsS = append_label(class_num, 0.0 if raid["model"][i] == "human" else 1.0)
    elif class_num == 2:
        labelsL = append_label(class_num, 0.0 if raid["model"][i] == "human" else 1.0)

    # Handle responses based on classification
    if class_num == 0:
        print("Calling models academic")
        acad, sci, lit = process_responses((acadPrompt, sciPrompt, litPrompt), generation)
        if acad == -1.0 or sci == -1.0 or lit == -1.0:
            labelsA = labelsA[:-1]
            continue
        acadResponsesA = np.append(acadResponsesA, acad)
        sciResponsesA = np.append(sciResponsesA, sci)
        litResponsesA = np.append(litResponsesA, lit)
        print("appended responses")
    elif class_num == 1:
        print("Calling models scientific")
        acad, sci, lit = process_responses((acadPrompt, sciPrompt, litPrompt), generation)
        if acad == -1.0 or sci == -1.0 or lit == -1.0:
            labelsS = labelsS[:-1]
            continue
        acadResponsesS = np.append(acadResponsesS, acad)
        sciResponsesS = np.append(sciResponsesS, sci)
        litResponsesS = np.append(litResponsesS, lit)
        print("appended responses")
    elif class_num == 2:
        print("Calling models literature")
        acad, sci, lit = process_responses((acadPrompt, sciPrompt, litPrompt), generation)
        if acad == -1.0 or sci == -1.0 or lit == -1.0:
            labelsL = labelsL[:-1]
            continue
        acadResponsesL = np.append(acadResponsesL, acad)
        sciResponsesL = np.append(sciResponsesL, sci)
        litResponsesL = np.append(litResponsesL, lit)
        print("appended responses")


Xa = np.column_stack((acadResponsesA, sciResponsesA, litResponsesA))
print(len(Xa))
modelA = LinearRegression()
modelA.fit(Xa, labelsA)
weightsA = modelA.coef_
interceptA = modelA.intercept_
print(interceptA)
print(weightsA)
print()


Xs = np.column_stack((acadResponsesS, sciResponsesS, litResponsesS))
print(len(Xs))
modelS = LinearRegression()
modelS.fit(Xs, labelsS)
weightsS = modelS.coef_
interceptS = modelS.intercept_
print(interceptS)
print(weightsS)
print()


Xl = np.column_stack((acadResponsesL, sciResponsesL, litResponsesL))
print(len(Xl))
modelL = LinearRegression()
modelL.fit(Xl, labelsL)
weightsL = modelL.coef_
interceptL = modelL.intercept_
print(interceptL)
print(weightsL)
print()
