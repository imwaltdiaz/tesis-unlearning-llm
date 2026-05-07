import os
import json


def combine_previous_test_jsons(output_dir, current_time_step_num):
    combined_test_qa_pairs = []

    # Iterate over the previous time steps to gather all test QA pairs
    for time_step_num in range(1, current_time_step_num):
        previous_json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}',
                                               f'time_step_{time_step_num}_test_dataset_unlearn.json')

        # Check if the file exists before attempting to open it
        if os.path.exists(previous_json_file_path):
            with open(previous_json_file_path, 'r', encoding='utf-8') as json_file:
                test_qa_pairs = json.load(json_file)
                combined_test_qa_pairs.extend(test_qa_pairs)
        else:
            print(f"Warning: {previous_json_file_path} does not exist and will be skipped.")

    # Define the output JSON file path for the combined test data
    combined_json_file_path = os.path.join(output_dir, f'time_step_{current_time_step_num}',
                                           f'time_step_{current_time_step_num}_combined_previous_tests.json')

    # Write the combined list of test QA pairs to a JSON file
    with open(combined_json_file_path, 'w', encoding='utf-8') as combined_json_file:
        json.dump(combined_test_qa_pairs, combined_json_file, ensure_ascii=False, indent=4)

    print(f"Combined test JSON file for time_step_{current_time_step_num} has been saved to {combined_json_file_path}")


# Loop through time steps starting from time_step_num > 1
for time_step_num in range(2, 11):
    output_dir = 'data_csv_single'
    combine_previous_test_jsons(output_dir, time_step_num)
