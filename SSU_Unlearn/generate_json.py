import pandas as pd
import json
import os

def convert_csv_to_json(csv_file_path, output_dir, time_step_num):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Shuffle the DataFrame with a fixed seed and select the top 200 examples
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(200)

    # Create a list of dictionaries, where each dictionary represents a question-answer pair
    qa_pairs = []
    for index, row in df.iterrows():
        qa_pairs.append({
            "question": row['question'],
            "answer": row['answer']
        })

    # Define the output JSON file path
    json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}_combined_dataset_unlearn.json')

    # Write the list of dictionaries to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)

    print(f"JSON file has been saved to {json_file_path}")

# time_step_num = 1
# csv_file_path = f'data_ga_csv/time_step_{time_step_num}/time_step_{time_step_num}_combined_dataset_unlearn.csv'
# output_dir = f'data_ga_csv/time_step_{time_step_num}'
# convert_csv_to_json(csv_file_path, output_dir, time_step_num)
time_step_num = 1
for time_step_num in range(1, 11):
    csv_file_path = f'data_csv_single/time_step_{time_step_num}/time_step_{time_step_num}_combined_dataset_unlearn.csv'
    output_dir = f'data_csv_single/time_step_{time_step_num}'
    convert_csv_to_json(csv_file_path, output_dir, time_step_num)