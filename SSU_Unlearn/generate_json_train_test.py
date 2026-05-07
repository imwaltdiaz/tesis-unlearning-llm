import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def convert_csv_to_json(csv_file_path, output_dir, time_step_num, mode):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Shuffle the DataFrame with a fixed seed and split into train and test sets
    if mode == "train":
        train_df, test_df = train_test_split(df, test_size=0, random_state=42)

        # Create lists of dictionaries for train and test data
        train_qa_pairs = [
            {"question": row['question'], "answer": row['answer']}
            for index, row in train_df.iterrows()
        ]
        test_qa_pairs = [
            {"question": row['question'], "answer": row['answer']}
            for index, row in test_df.iterrows()
        ]

        # Define the output JSON file paths for train and test
        train_json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}_train_dataset_unlearn.json')
        test_json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}_test_dataset_unlearn.json')

        # Write the train and test lists of dictionaries to JSON files
        with open(train_json_file_path, 'w', encoding='utf-8') as train_json_file:
            json.dump(train_qa_pairs, train_json_file, ensure_ascii=False, indent=4)
        with open(test_json_file_path, 'w', encoding='utf-8') as test_json_file:
            json.dump(test_qa_pairs, test_json_file, ensure_ascii=False, indent=4)

        print(f"Train JSON file has been saved to {train_json_file_path}")
        print(f"Test JSON file has been saved to {test_json_file_path}")

    else:
        # Create a list of dictionaries for all data
        qa_pairs = [
            {"question": row['question'], "answer": row['answer']}
            for index, row in df.iterrows()
        ]

        # Define the output JSON file path
        json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}_combined_dataset_unlearn.json')

        # Write the list of dictionaries to a JSON file
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)

        print(f"JSON file has been saved to {json_file_path}")


# Loop through time steps and specify the mode
for time_step_num in range(1, 11):
    csv_file_path = f'data_csv_single/time_step_{time_step_num}/time_step_{time_step_num}_combined_dataset_unlearn.csv'
    output_dir = f'data_csv_single/time_step_{time_step_num}'

    mode = "train"
    
    convert_csv_to_json(csv_file_path, output_dir, time_step_num, mode)
