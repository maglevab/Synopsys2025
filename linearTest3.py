import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests



master = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/LinRegData/masterDatasetv2.csv")
datasetLength = 414
df = master.sample(n = datasetLength, random_state = 50, ignore_index = True)

#Science --> 0
#Social Media --> 1
#Literature --> 2
classPrompt = "Remember examples of Academic Text such as essays or school assignments and examples of scientific texts such as research papers and abstracts as “category 0 text”, Remember examples of Social media texts such as news reports, reddit threads, reviews, and wikipedia entries as “Category 1 text”, Remember examples of Fiction such as novels or stories as “Category 2 text”. using these categories, analyse the given text and output the best fit category, do not add reasoning, but only the string containing the word category and the number. Here is the text: "
sciPrompt = "You are a model that specializes in detecting AI-generated academic and  scientific texts. This means scientific or research papers or abstract, but also essays and school assignments of any subject. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "
smPrompt = "You are a model that specializes in detecting AI-generated texts from social media. This means news articles, reddit posts, reviews of products, and wikipedia entries. For your output, remember that the number 1 represents ai-generated text and the number 0 represents non ai-generated text. I will give you a piece of text and I want you to detect whether it is ai-generated or not. For your output, only output the word 'class' followed by one of the two numbers, do not add any reasoning. Here is your text: "
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


smResponsesM = np.array([])
smResponsesS = np.array([])
smResponsesL = np.array([])
sciResponsesM = np.array([])
sciResponsesS = np.array([])
sciResponsesL = np.array([])
litResponsesM = np.array([])
litResponsesS= np.array([])
litResponsesL = np.array([])
labelsM = np.array([])
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
            print("Max retries reached.")
            return -1  # Default class


def append_label(class_num, true_answer):
    """Append labels based on classification."""
    if class_num == 0:
        print("Appending to labelsS")
        return np.append(labelsS, true_answer)
    elif class_num == 1:
        print("Appending to labelsM")
        return np.append(labelsM, true_answer)
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
    sm, sci, lit = prompts
    sm_response = call_model(sm, generation)
    sci_response = call_model(sci, generation)
    lit_response = call_model(lit, generation)
    return sci_response, sm_response, lit_response


for i, generation in enumerate(df["text"]):
    print(i)
    print(generation)

    # Classify data
    class_num = classify_data(classPrompt, generation)
    print(class_num)
    if class_num == -1:
        continue
    if class_num == 0:
        labelsS = append_label(class_num, df["label"][i])
    elif class_num == 1:
        labelsM = append_label(class_num, df["label"][i])
    elif class_num == 2:
        labelsL = append_label(class_num, df["label"][i])

    # Handle responses based on classification
    if class_num == 0:
        print("Calling models scientific")
        sm, sci, lit = process_responses((smPrompt, sciPrompt, litPrompt), generation)
        if sm == -1.0 or sci == -1.0 or lit == -1.0:
            labelsS = labelsS[:-1]
            continue
        smResponsesS = np.append(smResponsesS, sm)
        sciResponsesS = np.append(sciResponsesS, sci)
        litResponsesS = np.append(litResponsesS, lit)
        print("appended responses")
    elif class_num == 1:
        print("Calling models social media")
        sm, sci, lit = process_responses((smPrompt, sciPrompt, litPrompt), generation)
        if sm == -1.0 or sci == -1.0 or lit == -1.0:
            labelsS = labelsS[:-1]
            continue
        smResponsesM = np.append(smResponsesM, sm)
        sciResponsesM = np.append(sciResponsesM, sci)
        litResponsesM = np.append(litResponsesM, lit)
        print("appended responses")
    elif class_num == 2:
        print("Calling models literature")
        sm, sci, lit = process_responses((smPrompt, sciPrompt, litPrompt), generation)
        if sm == -1.0 or sci == -1.0 or lit == -1.0:
            labelsL = labelsL[:-1]
            continue
        smResponsesL = np.append(smResponsesL, sm)
        sciResponsesL = np.append(sciResponsesL, sci)
        litResponsesL = np.append(litResponsesL, lit)
        print("appended responses")


np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/smResponsesS", smResponsesS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/sciResponsesS", sciResponsesS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/litResponsesS", litResponsesS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/smResponsesM", smResponsesM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/sciResponsesM", sciResponsesM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/litResponsesM", litResponsesM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/smResponsesL", smResponsesL)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/sciResponsesL", sciResponsesL)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/litResponsesL", litResponsesL)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/labelsS", labelsS)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/labelsM", labelsM)
np.save("/Users/arahan/Desktop/Synopsys 2025/npArrays/labelsL", labelsL)

min_len = min(len(smResponsesS), len(sciResponsesS), len(litResponsesS))
smResponsesS = smResponsesS[:min_len]
sciResponsesS = sciResponsesS[:min_len]
litResponsesS = litResponsesS[:min_len]
labelsS = labelsS[:min_len]

Xs = np.column_stack((smResponsesS, sciResponsesS, litResponsesS))
print(len(Xs))
modelS = LinearRegression()
modelS.fit(Xs, labelsS)
weightsS = modelS.coef_
interceptS = modelS.intercept_
print(interceptS)
print(weightsS)
print()

min_len = min(len(smResponsesM), len(sciResponsesM), len(litResponsesM))
smResponsesM = smResponsesM[:min_len]
sciResponsesM = sciResponsesM[:min_len]
litResponsesM = litResponsesM[:min_len]
labelsM = labelsM[:min_len]

Xm = np.column_stack((smResponsesM, sciResponsesM, litResponsesM))
print(len(Xm))
modelM = LinearRegression()
modelM.fit(Xm, labelsM)
weightsA = modelM.coef_
interceptA = modelM.intercept_
print(interceptA)
print(weightsA)
print()

min_len = min(len(smResponsesL), len(sciResponsesL), len(litResponsesL))
smResponsesL = smResponsesL[:min_len]
sciResponsesL = sciResponsesL[:min_len]
litResponsesL = litResponsesL[:min_len]
labelsL = labelsL[:min_len]

Xl = np.column_stack((smResponsesL, sciResponsesL, litResponsesL))
print(len(Xl))
modelL = LinearRegression()
modelL.fit(Xl, labelsL)
weightsL = modelL.coef_
interceptL = modelL.intercept_
print(interceptL)
print(weightsL)
print()
