import pandas as pd

newsOld = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv3.csv")

news = { "text": [],
               "label": []}
news = pd.DataFrame(news)
for i in range(len(newsOld)):
    text = newsOld["text"][i]
    label = newsOld["label"][i]

    newText = text[:1000]
    newText = newText.replace('\n', ' ')
    news.loc[i] = [newText, label]


news.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv4.csv")