import requests
import json

url = "http://localhost:11434/api/chat"

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


texts =["The novel Beloved examines the approaches to dealing with the past through Sethe’s unwillingness to let go of the past, Ella’s confrontation of the past, and Paul D’s locking away of the past, to show the varied effects the past has on shaping the present. The novel illustrates how the past should not take the reins of the present, and could be transformed into a source of resilience and redefinition, only if it is confronted and understood, not merely buried or relived.",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way.",
        "Mars is half Earth's size and much colder, but its arid surface looks oddly familiar, with rocky plains, rolling hills, and sand dunes much like those on Earth. THe dusty ground it tinged brownish red by rust (iron oxide) and makes Mars look reddish from Earth, which is why the ancient Greeks and Romans named the planet after their god of war. Mars may have been warmer and wetter in the past, and there are signs that water once flowed across its surface carving out gullies and laying down sedimentary rock.",
        "Reproductive rights groups are preparing for legal battles in several of the states where voters approved constitutional amendments to protect or expand abortion access this month. After seven of the 10 pro-abortion rights measures on the November ballot across the country passed, the groups that backed them are now working to ensure they are implemented smoothly, particularly in states where the amendments will undo existing abortion bans.",
        "Our empirical evaluations across multiple datasets (Xsum, Squad, IMDb, and Kaggle FakeNews) confirm the viability of enhanced detection methods. We test various state-of-the-art text generators, including GPT-2, GPT-3.5-Turbo, Llama, Llama-2-13B-Chat-HF, and Llama-2-70B-Chat-HF, against detectors, including oBERTa-Large/Base-Detector, GPTZero. Our findings align with OpenAI's empirical data related to sequence length, marking the first theoretical substantiation for these observations.",
        "Dear Mr. Davis, I’m Sarah Mitchell, the Head of Development at EcoWrap Solutions. We specialize in biodegradable packaging, and I believe our products align well with GreenFlow Logistics' sustainability goals. Our latest packaging line reduces waste by 40% and is cost-efficient. I’d love to schedule a 15-minute call to discuss how we could collaborate. Are you available next Tuesday at 2 PM? Looking forward to your thoughts! Best regards, Sarah Mitchell"
         ]
for i in range(6):

    text = texts[i]
    response = llama3("I will give you a lengthy piece of text as an input, and I want you to classify this text into 1 of 6 different options. "
                  "0 = Academic Text such as essays, 1 = Fiction such as novels or stories, "
                  "2 = Non-Fiction such as memoirs, 3 = News Articles or Journalism such as articles or columns, "
                  "4 = Scientific texts such as research papers, and 5 = Business Texts such as emails, reports, or proposals. "
                  "Here is the text: " + text + ". Please output only one number.")

    print(response)