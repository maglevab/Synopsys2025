import pandas as pd
from keras._tf_keras.keras.models import load_model

distilBertModel = load_model("/Users/arahan/Downloads/distil_bert-keras-distil_bert_base_en_uncased-v3/model.weights.h5")
huggingFaceModel = load_model("/Users/arahan/Desktop/Synopsys 2025/Models/HuggingFaceModel.h5")

raid = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv2.csv")
raid = raid[:600]

experts = [distilBertModel, huggingFaceModel]
weights = [1]*len(experts)
labels = raid["domain"]
texts = raid["generation"]
weightTuner = 0.75

def get_result(model, text):
    #takes the model and gets a result
    pass

#running the weights
for i in range(len(raid)):
    #iterate through all the models and get their response
    answers = []
    for j in range(len(experts)):
        answers.append(get_result(experts[j], texts[i]))

    for j in range(len(answers)):
        if answers[j] != labels[i]:
            weights[j] *= weightTuner

print(weights)


