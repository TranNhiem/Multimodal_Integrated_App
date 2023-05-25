import opencc
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Function to convert text from simplified to traditional Chinese
def convert_simplified_to_traditional(text):
    converter = opencc.OpenCC('s2t.json')
    return converter.convert(text)

file_path = "/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Instruction_finetune_dataset/Belle_open_source_0.5M.json"

def load_input_data(INPUT_TASKS_PATH):
    with open(INPUT_TASKS_PATH, 'r') as json_file:
        content = json_file.read()

    # Split the content by line and remove empty lines
    json_objects = [line for line in content.splitlines() if line.strip()]

    df_list = []  # List to store DataFrames for each JSON object

    # Iterate through the JSON objects, load and convert them into DataFrames
    for index, json_object in enumerate(json_objects):
        try:
            data = json.loads(json_object)
            df = pd.DataFrame([data], index=[index])  # Convert JSON object to DataFrame with index
            df_list.append(df)  # Append DataFrame to list
        except (json.JSONDecodeError, ValueError) as err:
            print(f"Error parsing JSON Object {index + 1}: {err}")

    # Concatenate the DataFrames in the list into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)
    print(f"Complete Loaded {len(final_df)} JSON objects.")
    return final_df


input_data = load_input_data(file_path)

# Convert the JSON data in parallel
def convert_text(df, start, end, subset=True, output_file="traditional_chinese.json"):
    if subset:
        subset_df = df.iloc[start:end]#.copy()
    else:
        subset_df = df#.copy()
    
    #original_subset_df = subset_df.copy()  # Make a copy of the original subset DataFrame


    instructions = subset_df['instruction'].tolist()
    inputs = subset_df['input'].tolist()
    outputs = subset_df['output'].tolist()

    with ThreadPoolExecutor() as executor:
        instructions = list(executor.map(convert_simplified_to_traditional, instructions))
        inputs = list(executor.map(convert_simplified_to_traditional, inputs))
        outputs = list(executor.map(convert_simplified_to_traditional, outputs))

    subset_df['instruction'] = instructions
    subset_df['input'] = inputs
    subset_df['output'] = outputs

    # Save the converted DataFrame to a JSON file
    subset_df.to_json(output_file, orient='records', force_ascii=False, indent=4)
    # Save the original and converted DataFrames to separate files
    # original_file = "original_data.json"
    # original_subset_df.to_json(original_file, orient='records', force_ascii=False)

    return subset_df

# Example usage:
start = 0
end = 10
output_file = "converted_Traditional_chinese_Belle_open_source_0.5M.json"
converted_data = convert_text(input_data, start, end,subset=False, output_file=output_file)
