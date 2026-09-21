from transformers import AutoModelForCausalLM, AutoTokenizer

modelName = "llama3"
tokenizer = AutoTokenizer.from_pretrained(modelName)
model= AutoModelForCausalLM.from_pretrained(modelName)