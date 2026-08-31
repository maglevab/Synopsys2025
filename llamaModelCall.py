import requests

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

def classify_data(generation):
    """Classify the data using the model and handle retries."""
    while_count = 0
    while True:
        class_num = llamaModel(classPrompt + generation)
        class_num = class_num[9]
        if class_num in {"0", "1", "2"}:
            return int(class_num)
        print("Redoing classification...")
        while_count += 1
        if while_count == 10:
            print("Max retries reached.")
            return -1  # Default class

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