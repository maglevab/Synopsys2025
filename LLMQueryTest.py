from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/Users/arahan/.llama/checkpoints/Llama3.1-8B"


tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device="auto")

input_text = "Generate 10 random letters"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(inputs.input_ids, max_length=50, temperature=0.7, num_return_sequences=1)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

print(generated_text)