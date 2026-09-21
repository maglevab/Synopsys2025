import numpy as np
import pandas as pd
import joblib
import expertModelCall, llamaModelCall
from sklearn.model_selection import train_test_split

#load in test dataset
academic = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/abstractsScientificv2.csv")
academicSample, _ = train_test_split(academic, train_size=30, stratify=academic["label"], random_state=5)
news = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv4.csv")
newsSample, _ = train_test_split(news, train_size=30, stratify=news['label'], random_state=5)
litH = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDatasetHUMAN.csv")
litHSample, _ = train_test_split(litH, train_size=15, random_state=5)
litAI = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/litDataAI.csv")
litAISample, _ = train_test_split(litAI, train_size=15, random_state=5)

test = pd.concat([academicSample, newsSample, litHSample, litAISample], ignore_index=True)
print(test.head())
print(len(test))
testLength=len(test)
classes = [0, 0, 0]
#load in compilers
AcademicCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModels/AcademicCompiler.pkl")
MediaCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModels/MediaCompiler.pkl")
LiteratureCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModels/LiteratureCompiler.pkl")

def runFlow(text):

    #classify text
    print("classifying text")
    classNum = llamaModelCall.classify_data(text)
    classes[classNum] += 1
    print("classified as " + str(classNum))
    #query expert models
    print("querying expert models")
    academicAnswer = expertModelCall.predict(text, "acad_expert")
    mediaAnswer = expertModelCall.predict(text, "news_expert")
    literatureAnswer = expertModelCall.predict(text, "lit_expert")
    print("compiling answers")
    #use compiler
    answers = np.array([[academicAnswer, mediaAnswer, literatureAnswer]])
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

    return finalAnswer
correct = 0
for i in range(testLength):
    text = test["text"][i]
    finalAnswer = runFlow(text)
    if finalAnswer == -1:
        continue
    if finalAnswer == test["label"][i]:
        correct += 1
    print(finalAnswer)

print(correct)
print(testLength)
print(correct/testLength)
print(classes)