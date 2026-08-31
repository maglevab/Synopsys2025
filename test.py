import pandas as pd

df = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/trainRandomv2.csv")
train_data = pd.read_csv("/Users/arahan/Desktop/Synopsys 2025/Data/abstractsScientific.csv")
print(train_data["label"].value_counts())
train0 = train_data[train_data["label"] == 0]
train1 = train_data[train_data["label"] == 1]

train0 = train0.iloc[:3000]
train1 = train1.iloc[:3000]

df2 = df[df["domain"].isin(["abstracts", "books", "news", "poetry", "wiki", "reddit", "reviews"])]
df2 = df2[["domain", "generation", "model"]]

df2Abstracts = df2[df2["domain"] == "abstracts"]
df2Books = df2[df2["domain"] == "books"]
df2News = df2[df2["domain"] == "news"]
df2Poetry = df2[df2["domain"] == "poetry"]
df2Wiki = df2[df2["domain"] == "wiki"]
df2Reviews = df2[df2["domain"] == "reviews"]
df2Reddit = df2[df2["domain"] == "reddit"]

dfA_S = df2Abstracts
dfL = pd.concat([df2Books, df2Poetry])
dfN_SM = pd.concat([df2News, df2Wiki, df2Reviews, df2Reddit])

print("length of academic and science is " + str(len(dfA_S)))
print("number of human generated is " + str((dfA_S["model"] == "human").sum()))
print("number of ai generated is " + str((len(dfA_S)-(dfA_S["model"] == "human").sum())))
print()
print("length of literature is " + str(len(dfL)))
print("number of human generated is " + str((dfL["model"] == "human").sum()))
print("number of ai generated is " + str((len(dfL)-(dfL["model"] == "human").sum())))
print()
print("length of news and social media is " + str(len(dfN_SM)))
print("number of human generated is " + str((dfN_SM["model"] == "human").sum()))
print("number of ai generated is " + str((len(dfN_SM)-(dfN_SM["model"] == "human").sum())))