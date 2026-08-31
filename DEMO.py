import joblib
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
import llamaModelCallv2
import numpy as np

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

def predict_text(model, text):
    global tokenizer
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class#, probabilities

def runFlow(text):

    #classify text
    classNum = llamaModelCallv2.classify_data(text)
    #query expert models
    academicAnswer = predict_text(academicExpert, text)
    mediaAnswer = predict_text(mediaExpert, text)
    literatureAnswer = predict_text(literatureExpert, text)
    #use synthesizer
    answers = np.array([[academicAnswer, mediaAnswer, literatureAnswer]])
    if classNum == 0:
        finalAnswer = max(AcademicCompiler.predict(answers))
    elif classNum == 1:
        finalAnswer = max(MediaCompiler.predict(answers))
    elif classNum == 2:
        finalAnswer = max(LiteratureCompiler.predict(answers))
    else:
        print("Not able to classify text, please check your input.")
        finalAnswer = -1

    return finalAnswer, classNum

print("Hello, welcome to AuthenText.\nAuthenText is a universal application for detecting AI-Generated text developed by Arahan Balasubramanian.")
text = input("Please enter a text: ")

finalAnswer, classNum = runFlow(text)
if classNum == 0:
    print("This text is classified as Academic/Scientific Text")
elif classNum == 1:
    print("This text is classified as News/Social Media Text")
elif classNum == 2:
    print("This text is classified as Literature Text")
print("The text is classified as " + str(classNum))
if finalAnswer == 0:
    print("The text is found to be Human generated")
elif finalAnswer == 1:
    print("The text is found to be AI generated")