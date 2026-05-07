# scripts/02_prepare_datasets.py
import json
import os
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def get_tokenizer(model_name: str = MODEL_NAME):
    """Lazy tokenizer loader.

    Importar este módulo para usar FORGET_BOOKS/RETAIN_BOOKS no debería
    disparar descargas ni consumir memoria.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def clean_gutenberg_text(text):
    """Elimina headers/footers de Project Gutenberg"""
    # Busca inicio real del libro
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
    ]
    
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Avanza hasta el siguiente salto de línea doble
            start_idx = text.find('\n\n', idx) + 2
            break
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
    
    return text[start_idx:end_idx].strip()

def create_chunks(text, chunk_size=200, prompt_size=100, tokenizer=None):
    """Divide texto en chunks de chunk_size tokens"""
    tokenizer = tokenizer or get_tokenizer()
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens) - chunk_size, chunk_size):
        chunk = tokens[i:i + chunk_size]
        prompt_tokens = chunk[:prompt_size]
        answer_tokens = chunk[prompt_size:]
        
        chunks.append({
            "input": tokenizer.decode(prompt_tokens),
            "output": tokenizer.decode(answer_tokens),
            "full": tokenizer.decode(chunk)
        })
    
    return chunks

def prepare_book_dataset(book_path, book_name, output_dir, tokenizer=None):
    """Prepara dataset completo de un libro"""
    tokenizer = tokenizer or get_tokenizer()
    with open(book_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    text = clean_gutenberg_text(text)
    chunks = create_chunks(text, tokenizer=tokenizer)
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/{book_name}.json"
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"{book_name}: {len(chunks)} chunks generados → {output_path}")
    return chunks

def build_all_processed_books(
    books_dir: str = "data/raw/gutenberg",
    output_dir: str = "data/processed/gutenberg",
    model_name: str = MODEL_NAME,
):
    """Procesa todos los libros raw -> JSON procesado."""
    tokenizer = get_tokenizer(model_name)
    for filename in os.listdir(books_dir):
        if filename.endswith(".txt"):
            book_name = filename.replace(".txt", "")
            prepare_book_dataset(
                f"{books_dir}/{filename}",
                book_name,
                output_dir,
                tokenizer=tokenizer,
            )


## DEFINE QUE DESAPRENDER
# La partición que usarás en tu tesis
FORGET_BOOKS = [
    "sherlock_holmes",      # paso 1
    "pride_prejudice",      # paso 2  
    "alice_wonderland",     # paso 3
    "moby_dick",            # paso 4
    "great_gatsby",         # paso 5
]

RETAIN_BOOKS = [
    "war_peace",
    "tale_two_cities", 
    "frankenstein",
    "dracula",
    "adventures_huck_finn",
]

# Para desaprendizaje SECUENCIAL en lotes (tu aportación)
# Divide cada libro en 3 lotes temporales
def create_sequential_batches(book_chunks, n_batches=3):
    """Tu contribución: partición iterativa de D_forget"""
    batch_size = len(book_chunks) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size if i < n_batches - 1 else len(book_chunks)
        batches.append(book_chunks[start:end])
    return batches


if __name__ == "__main__":
    build_all_processed_books()