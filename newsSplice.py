import pandas as pd

df = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv3.csv")


# Function to split text into 1000-character chunks
def split_text(text, label, chunk_size=1000):
    return [(text[i:i+chunk_size], label) for i in range(0, len(text), chunk_size)]

# Apply function to each row and flatten the list
new_data = []
for _, row in df.iterrows():
    new_data.extend(split_text(row['text'], row['label']))

# Create new DataFrame
new_df = pd.DataFrame(new_data, columns=['text', 'label'])

# Save or use the new dataset
new_df.to_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv5.csv", index=False)