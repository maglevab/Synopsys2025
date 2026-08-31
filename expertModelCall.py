from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Define model paths
model_paths = {
    "news_expert": "/Users/arahan/Desktop/Synopsys2025/ExpertModels/newsExpertModel",
    "lit_expert": "/Users/arahan/Desktop/Synopsys2025/ExpertModels/litExpertModel",
    "acad_expert": "/Users/arahan/Desktop/Synopsys2025/ExpertModels/AcademicModel"
}

# Load tokenizers (assuming different architectures may be used)
tokenizers = {
    "news_expert": AutoTokenizer.from_pretrained(model_paths["news_expert"]),
    "lit_expert": AutoTokenizer.from_pretrained(model_paths["lit_expert"]),
    "acad_expert": AutoTokenizer.from_pretrained(model_paths["acad_expert"])
}

# Load multiple models
models = {
    name: AutoModelForSequenceClassification.from_pretrained(path).eval()
    for name, path in model_paths.items()
}

# Preprocessing function (chooses the correct tokenizer)
def preprocess_text(text, model_name):
    tokenizer = tokenizers[model_name]
    return tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length = 512)

def process_responses(model_names, text):
    """Process responses for a given set of prompts."""
    sm, sci, lit = model_names
    sm_response = predict(text, sm)
    sci_response = predict(text, sci)
    lit_response = predict(text, lit)
    return sci_response, sm_response, lit_response


# Prediction function
def predict(text, model_name):
    inputs = preprocess_text(text, model_name)

    with torch.no_grad():
        outputs = models[model_name](**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

    return np.argmax(probs)