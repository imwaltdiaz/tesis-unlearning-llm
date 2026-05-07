from dataportraits import datasketch
import dataportraits.utils as utils
import subprocess
import pandas as pd
import os
import csv
from transformers import AutoTokenizer

#  ps aux | grep redis-server
# Connect to Redis
subprocess.run(f"python easy_redis.py --shutdown", shell=True, check=True, capture_output=True)
subprocess.run(f"python easy_redis.py --just-start", shell=True, check=True, capture_output=True)

def count_tokens(string_list):
    total_tokens = 0
    for string in string_list:
        tokens = string.split()  # Split the string into tokens by whitespace
        total_tokens += len(tokens)
    return total_tokens

def main(args):
    datatype = args.datatype
    n=args.n
    width=n
    stride=n
    time_step_num = args.time_step
    if datatype != 'gutenberg_books':
        if args.tokenized:
            bf_name = f'{datatype}_tokenized.{6*width}-{6*stride}.bf'
        else:
            bf_name = f'{datatype}.{width}-{stride}.bf'
    else:
        if args.tokenized:
            bf_name = f'{datatype}_time_step_{time_step_num}_tokenized.{6*width}-{6*stride}.bf'
        else:
            bf_name = f'{datatype}_time_step_{time_step_num}.{width}-{stride}.bf'

    tokenizer_path = 'meta-llama/Meta-Llama-3.1-8B'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    r = datasketch.RedisBFSketch(host='localhost', port=6379, key=bf_name, width=int(n))
    r.bf_client.create(bf_name, 0.001, 1500000000) # bfname, error_rate, max_entries


    if datatype == 'newsqa':
        path = '../eval_data/newsqa/newsqa_blocklisted_infringement.csv'
    elif datatype == 'booksum':
        path = '../eval_data/booksum/booksum_blocklisted_infringement.csv'
    elif datatype == 'gutenberg_books':
        path = f'./SSU_Unlearn/data_csv_single/time_step_{time_step_num}/time_step_{time_step_num}_combined_dataset_bf.csv'


    # List of lyrics

    if datatype in ['newsqa']:
        newsqa_dataset = pd.read_csv(path)
        input_sequences  = newsqa_dataset['story_text']
    elif datatype in ['booksum']:
        booksum_dataset = pd.read_csv(path)
        input_sequences = booksum_dataset['document'].to_list()
        print(len(input_sequences))
    elif datatype in ['gutenberg_books']:
        gutenberg_dataset = pd.read_csv(path)
        input_sequences = gutenberg_dataset["document"].to_list()
   
    print(f"successfully loaded testing chunks from {path}")

    if args.tokenized:  
        text_pipeline=datasketch.build_text_pipeline_fn(width=6*n, stride=6, apply_code_processor=True) # Here we transfer all token_id into a 6-digit number
        tokenized_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in input_sequences]
        print("tokenized_sequences's shape:", len(tokenized_sequences), len(tokenized_sequences[0]), "\n")
        tokenized_sequences_tostring=[''.join(f"{num:06d}" for num in sublist) for sublist in tokenized_sequences]
        print("tokenized_sequences_tostring's shape:", len(tokenized_sequences_tostring), len(tokenized_sequences_tostring[0]), "\n")
        # tokenized_sequences_tostring = [np.array(seq).tobytes() for seq in tokenized_sequences]
        grams=utils.flatten_batched(text_pipeline(batches_of_text=tokenized_sequences_tostring))
        r.redis_client.execute_command('BF.MADD', bf_name, *grams[1])
        # for gram in tqdm(grams[1]):
        #     token_ids = tokenizer.encode(gram, return_tensors='pt', add_special_tokens=False).squeeze().tolist()
        #     token_ids_hash = hash_token_ids(token_ids)
        
    else:
        text_pipeline=datasketch.build_text_pipeline_fn(width=n, stride=n, apply_code_processor=True)
        grams=utils.flatten_batched(text_pipeline(batches_of_text=input_sequences))
        r.redis_client.execute_command('BF.MADD', bf_name, *grams[1])

    print("current directory is", os.getcwd())

    # Define the base directory path
    base_dir = './SSU_Unlearn/CoTaEval/data-portraits/bloom_filters'

    # Determine the specific subdirectory based on the tokenized flag
    if datatype != 'gutenberg_books':
        if args.single_book:
            sub_dir = f'{datatype}_single_book_tokenized/{n}' if args.tokenized else f'{datatype}_single_book/{n}'
        else:
            sub_dir = f'{datatype}_tokenized/{n}' if args.tokenized else f'{datatype}/{n}'
    else:
        if args.single_book:
            sub_dir = f'{datatype}_single_book_time_step_{time_step_num}_tokenized/{n}' if args.tokenized else f'{datatype}_single_book_time_step_{time_step_num}/{n}'
        else:
            sub_dir = f'{datatype}_time_step_{time_step_num}_tokenized/{n}' if args.tokenized else f'{datatype}_time_step_{time_step_num}/{n}'

    # Create the full directory path
    full_dir_path = os.path.join(base_dir, sub_dir)

    # Create the directory if it doesn't exist
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)

    # Define the full file path
    file_path = os.path.join(full_dir_path, bf_name)

    # Save the file
    r.to_file(path=file_path, verbose=False)  # verbose: Show the progress.


if __name__ == "__main__":
    # example: python create_bf.py --datatype gutenberg_books --n 6 --tokenized --single_book  --time_step 1
    import argparse
    parser = argparse.ArgumentParser(description='Create a bloom filter for a dataset')
    parser.add_argument('--datatype', type=str, default='lyrics_popular', help='Type of the dataset')
    parser.add_argument('--n', type=int, default=50, help='Width of the bloom filter')
    parser.add_argument('--tokenized', action='store_true', help='Whether the input is tokenized')
    parser.add_argument('--time_step', type=int, default=1, help='Time step')
    parser.add_argument('--single_book', action='store_true', help='Whether to use a single book to unlearn sequentially')
    args = parser.parse_args()

    # assert if the args.datatype equals "gutenberg_books", then "time_step" must be specified
    if args.datatype == "gutenberg_books":
        assert args.time_step is not None, "Please specify the time step for gutenberg"

    main(args)
