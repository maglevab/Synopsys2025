import requests
import pandas as pd
import json

raid = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv2.csv")
#raid = raid[:50]
labels = raid['domain']
texts = raid['generation']
url = "http://localhost:11434/api/chat"
correct = 0
count = 0
whileCount = 0
amtTries = 5
specialCount = 0
labelToKey = {"abstracts": 4, "wiki": 2, "poetry": 1, "books" : 1, "news": 3}

#Uses the model to get a response
def llama3(prompt):
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

    response = requests.post(url, headers = headers, json = data)
    return response.json()['message']['content']

#Converst the domain keys into a number
def convertdomain(domain):
    if domain == "abstracts" or domain == "wiki" or domain == "poetry" or domain == "books" or domain == "news":
        return labelToKey[domain]
    return -1


for i in range(len(labels)):

    print("i:" + str(i))
    if convertdomain(labels[i]) == -1:
        continue

    text = texts[i]
    whileCount = 0
    response = ""
    while len(response) != 15:
        response = llama3('Remember examples of Academic Text such as essays or school assignments as “category 0 text”, Remember examples of Fiction such as novels or stories as “Category 1 text”, Remember examples of Non-Fiction such as memoirs or factual pieces of texts as “category 2 text”, Remember examples of Newspaper Articles or other magazine articles journalistic in nature as “category 3 text”, Remember examples of Scientific texts such as research papers or abstracts as “category 4 text”, Remember examples of Business Texts such as emails, reports, or proposals as “category 5 text” using these categories, analyse the given text and output the best fit category, do not add reasoning, but only the string containing the word category and the number  ' + text)
        print(response)
        whileCount += 1
        if whileCount == amtTries:
            response = "Category 9 Text"

    response = response[9]
    if int(response) == convertdomain(labels[i]):
        correct += 1
    if int(response) == 9:
        specialCount += 1
    count += 1



print("Amount Correct: " + str(correct))
print("Amount Incorrect because of TimeOut: " + str(specialCount))
print("Total Amount: " + str(count))
print("Total Percentage: "+ str(float(correct/count)*100.0))