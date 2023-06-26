
import json
import random
import os 
import json 
folder_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Instruction_finetune_dataset/Mixed_Vi_dataset"

# Initialize an empty list to store the data from all JSON files
data = []

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            data.extend(json_data)

# Shuffle the data
random.shuffle(data)

# Print the shuffled data
for item in data[0:20]:
    print(item)

with open("/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Instruction_finetune_dataset/mixture_dataset.json", 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False)
mixture_data="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Instruction_finetune_dataset/mixture_dataset.json"
with open(mixture_data, 'r') as file:
    json_data = json.load(file)
    

# Dictionary to store word count statistics
word_count_stats = {}

# Iterate over each string
for string in json_data:
    string= string.get("response", "")
    # Split the string into words
    response_words = string.split()

    # Count the number of words
    word_count = len(response_words)

    # Update the statistics dictionary
    if word_count in word_count_stats:
        word_count_stats[word_count] += 1
    else:
        word_count_stats[word_count] = 1

# Print the word count statistics
print("Word Count Statistics:")
for count, frequency in word_count_stats.items():
    print(f"{count} words: {frequency} strings")