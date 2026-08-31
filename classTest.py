import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import expertModelCall

master = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/LinRegData/masterDatasetv3.csv")
df = master.sample(frac=1, random_state = 50, ignore_index = True)
print(len(df))
#Science --> 0
#Social Media --> 1
#Literature --> 2
classPrompt = "Remember examples of Academic Text such as essays or school assignments and examples of scientific texts such as research papers and abstracts as “category 0 text”, Remember examples of Social media texts such as news reports, reddit threads, reviews, and wikipedia entries as “Category 1 text”, Remember examples of Fiction such as novels or stories as “Category 2 text”. using these categories, analyse the given text and output the best fit category, do not add reasoning, but only the string containing the word category and the number. Here is the text: "
url = "http://localhost:11434/api/chat"
classDF = {
    "class": []
}

classDF = pd.DataFrame(classDF)
count = 0
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

def classify_data(prompt, text):
    """Classify the data using the model and handle retries."""
    while_count = 0
    while True:
        class_num = llamaModel(prompt + text)
        class_num = class_num[9]
        if class_num in {"0", "1", "2"}:
            return int(class_num)
        print("Redoing classification...")
        while_count += 1
        if while_count == 10:
            print("Max retries reached.")
            return -1  # Default class


for i, text in enumerate(df["text"]):
    print(i)
    print(text)

    # Classify data
    class_num = classify_data(classPrompt, text)
    if class_num == -1:
        count += 1
    classDF.loc[i] = [class_num]
classDF.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/classDF.csv")
print(count)