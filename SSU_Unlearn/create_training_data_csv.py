import os
import csv
import re
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess_text(text):
    text = text.lower()
    text = text.replace('â', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-')
    text = ' '.join(text.split())  # Remove unnecessary whitespace characters
    return text


def remove_header(text):
    header_pattern = r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*'
    header_match = re.search(header_pattern, text)

    if header_match:
        start = header_match.end()
        return text[start:]
    else:
        raise ValueError("Header not found in the text file.")


def remove_footer(text):
    footer_pattern = r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*'
    footer_match = re.search(footer_pattern, text)

    if footer_match:
        end = footer_match.start()
        return text[:end]
    else:
        raise ValueError("Footer not found in the text file.")


def preprocess_gutenberg_text(text):
    try:
        text = remove_header(text)
    except ValueError as e:
        print(f"Error: {e}")

    try:
        text = remove_footer(text)
    except ValueError as e:
        print(f"Error: {e}")

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def generate_dataset(book_path, output_file):
    with open(book_path, 'r', encoding='utf-8') as file:
        book_text = file.read()

    book_text = preprocess_gutenberg_text(book_text)

    sentences = sent_tokenize(book_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    all_words = [word for sentence in tokenized_sentences for word in sentence]

    dataset = []
    index = 0

    while index + 200 <= len(all_words):
        question_words = all_words[index:index + 100]
        answer_words = all_words[index + 100:index + 200]

        question = ' '.join(question_words)
        answer = ' '.join(answer_words)

        question = preprocess_text(question)
        answer = preprocess_text(answer)

        dataset.append({
            'question': question,
            'answer': answer
        })

        index += 200

    # remove the first element in the dataset list
    dataset.pop(0)

    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['question', 'answer'])
        writer.writeheader()
        for data in dataset:
            writer.writerow(data)

    print(f"Dataset generated and saved to {output_file}")


def combine_csv_files(csv_files, output_file):
    combined_dataset = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                combined_dataset.append(row)

    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['question', 'answer'])
        writer.writeheader()
        for data in combined_dataset:
            writer.writerow(data)

    print(f"Combined dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QA dataset from text.')
    parser.add_argument('--input_dir', type=str, required=True, help='The path to the input directory')
    args = parser.parse_args()

    for subdir, _, files in os.walk(args.input_dir):
        # Extract the time step number from the directory name
        match = re.search(r'time_step_(\d+)', os.path.basename(subdir))
        if not match:
            print(f"Skipping directory {subdir} as it does not match the pattern 'time_step_'")
            continue

        time_step = match.group(1)
        csv_files = []
        for file_name in files:
            if file_name.endswith('.txt'):
                book_path = os.path.join(subdir, file_name)
                book_name = os.path.splitext(file_name)[0]
                output_file = os.path.join(subdir, f'{book_name}_train.csv')
                generate_dataset(book_path, output_file)
                csv_files.append(output_file)

        if csv_files:
            combined_output_file = os.path.join(subdir, f'time_step_{time_step}_combined_dataset_unlearn.csv')
            combine_csv_files(csv_files, combined_output_file)
