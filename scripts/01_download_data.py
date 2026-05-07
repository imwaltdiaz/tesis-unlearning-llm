# scripts/01_download_data.py
import requests
import os

# IDs de Project Gutenberg
BOOKS = {
    "sherlock_holmes": 1661,
    "pride_prejudice": 1342,
    "alice_wonderland": 11,
    "moby_dick": 2701,
    "war_peace": 2600,
    "great_gatsby": 64317,  # dominio público desde 2021
    "tale_two_cities": 98,
    "frankenstein": 84,
    "dracula": 345,
    "adventures_huck_finn": 76,
}

def download_book(book_id, title):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    fallback = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    os.makedirs("data/raw/gutenberg", exist_ok=True)
    path = f"data/raw/gutenberg/{title}.txt"
    
    if os.path.exists(path):
        print(f"Ya existe: {title}")
        return
    
    for url in [url, fallback]:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(r.text)
                print(f"Descargado: {title}")
                return
        except:
            continue
    print(f"ERROR: no se pudo descargar {title}")

for title, book_id in BOOKS.items():
    download_book(book_id, title)