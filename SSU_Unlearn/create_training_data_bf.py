import os
import csv
import re

def combine_csv_files(csv_files, output_file):
    combined_dataset = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Combine question and answer into a single document
                row['document'] = row['question'] + ' ' + row['answer']
                combined_dataset.append(row)

    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        fieldnames = ['question', 'answer', 'document']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in combined_dataset:
            writer.writerow(data)

    print(f"Combined dataset saved to {output_file}")

if __name__ == "__main__":
    input_dir = 'data_csv_single'  # Adjust this path if needed

    # Get the list of time step directories and sort them in order
    time_step_dirs = sorted([d for d in os.listdir(input_dir) if re.match(r'time_step_\d+', d)], key=lambda x: int(re.search(r'\d+', x).group()))

    previous_combined_files = []

    for i, time_step_dir in enumerate(time_step_dirs, start=1):
        combined_unlearn_file = os.path.join(input_dir, time_step_dir, f'time_step_{i}_combined_dataset_unlearn.csv')
        combined_bf_file = os.path.join(input_dir, time_step_dir, f'time_step_{i}_combined_dataset_bf.csv')

        if os.path.exists(combined_unlearn_file):
            previous_combined_files.append(combined_unlearn_file)

        # Combine the previous combined files into the new combined_bf file
        combine_csv_files(previous_combined_files, combined_bf_file)
