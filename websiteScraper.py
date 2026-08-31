import requests
from bs4 import BeautifulSoup
import pandas as pd
import textwrap


def scrape_and_save(url, output_csv, all_chunks):
    # Fetch the webpage content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return all_chunks

    # Parse the webpage content
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)  # Extract visible text

    # Split text into 1000-character chunks
    chunks = textwrap.wrap(text, 1000)
    all_chunks.extend(chunks)
    print(f"Scraped {len(chunks)} chunks from {url}. Total collected chunks: {len(all_chunks)}")

    return all_chunks


# Run in a loop until user enters -1
output_csv = "scraped_data.csv"
all_chunks = []
while True:
    url = input("Enter the URL to scrape (or -1 to exit): ")
    if url == "-1":
        break
    all_chunks = scrape_and_save(url, output_csv, all_chunks)

# Save to CSV after all URLs are processed
if all_chunks:
    df = pd.DataFrame({'Text Chunk': all_chunks})
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved {len(all_chunks)} chunks to {output_csv}")
