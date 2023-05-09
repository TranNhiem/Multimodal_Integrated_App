import os
import openai
import json
import pandas as pd
import numpy as np
import re
import glob

# Microsoft Azure Setup
openai.api_type = "azure"
openai.api_base = "https://sslgroupservice.openai.azure.com/"
openai.api_base = "https://docs-test-001.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Original OpenAI Setup
# openai.api_key = "" # replace with your key
# MODEL = "gpt-3.5-turbo"

input_tasks_path = "./data/alpaca_52k_instruction.json"

with open(input_tasks_path, "rb") as f:
    json_data = json.loads(f.read())
    df = pd.DataFrame(json_data)

TARGET_LANGUAGE = "Vietnamese" # e.g. "English", "German", "Spanish"

# Save the given data to a JSON file at the specified file path
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

# Check if the given text matches the specified regular expression
def matches_regex(regex, text):
    return bool(re.compile(regex).search(text))

# Check if the given text contains code
def contains_code(text):
    code_blacklist = ['&&', '||', '<html>', ';\n', 'SELECT']
    return (
        any(code_keyword in text for code_keyword in code_blacklist) or
        matches_regex(r'\w+\(\w*\) \{', text) or
        matches_regex(r'def \w+\(', text) or
        matches_regex(r'\[A-z]+\.[A-z]+', text) or
        matches_regex(r': [\w\.#]{1,12};', text) or
        matches_regex(r'<\/\w+>', text)
    )

# Check if the given text contains words
def contains_words(text):
    return matches_regex(r'[A-z]{3,}', text)

# Check if the given text is translatable
def is_translatable(text):
    if text == "":
        return True
    return (contains_code(text) is False) and contains_words(text)

# Translate individual columns (instruction, input, and output) of each chunk as a list
def translate_and_update_series(text_series):
    is_translatable_index = text_series.apply(lambda x: is_translatable(x) is False)
    text_list_source_language = text_series.tolist()
    text_series[is_translatable_index] = ""
    text_list = text_series.tolist()
    translated_list = translate_list_openai(text_list)

    if is_translatable_index.sum() > 0:
        for index, text_is_translatable in enumerate(is_translatable_index.tolist()):
            if text_is_translatable:
                translated_list[index] = text_list_source_language[index]
    return translated_list

# Create OpenAI prompt string for translation
def create_openai_prompt_string(text):
    if ' ' in text:
        return f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    else:
        return f'Please provide the {TARGET_LANGUAGE} translation for the following word: {text}'

# Create OpenAI message list for translation
def create_openai_message_list(text_list):
    return [None if text == "" else {"role": "user", "content": create_openai_prompt_string(text)} for text in text_list]

# Translate OpenAI message
def translate_openai_message(message):
    if message is None:
        return ""
    
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                engine=MODEL,
                prompt=[message], #messages=[message],
            )
        except:
            pass
    #return response["choices"][0]["message"]["content"].strip()
    return response.choices[0].text.strip()

# Translate a list of text using OpenAI
def translate_list_openai(text_list):
    message_list = create_openai_message_list(text_list)
    return [translate_openai_message(message) for message in message_list]

chunk_size = 5
output_dir = './data/output/'

# Translate the DataFrame and save translated chunks to individual JSON files
def translate_dataframe(df):
    os.makedirs(output_dir, exist_ok=True)
    number_of_chunks = df.shape[0] // chunk_size
    chunked_df_list = np.array_split(df, number_of_chunks)
    
    start_index = 1
    
    for index, chunk_df in enumerate(chunked_df_list[start_index:]):
        instruction_list_translated = translate_and_update_series(chunk_df.instruction)
        input_list_translated = translate_and_update_series(chunk_df.input)
        output_list_translated = translate_and_update_series(chunk_df.output)
        
        translated_df = pd.DataFrame({'instruction': instruction_list_translated, 'input': input_list_translated, 'output': output_list_translated})
        translated_dict = translated_df.to_dict('records')
        
        write_json_file(translated_dict, f'{output_dir}chunk{start_index+index}.json')

# translate_dataframe(df)

# Translate a subset of the DataFrame and save translated chunks to individual JSON files
def translate_dataframe_subset(df, subset_size):
    os.makedirs(output_dir, exist_ok=True)

    # Extract the subset of the DataFrame
    subset_df = df.head(subset_size)

    # Translate the instructions, inputs, and outputs of the subset
    instruction_list_translated = translate_and_update_series(subset_df.instruction)
    input_list_translated = translate_and_update_series(subset_df.input)
    output_list_translated = translate_and_update_series(subset_df.output)

    # Create a DataFrame with the translated content
    translated_df = pd.DataFrame({'instruction': instruction_list_translated, 'input': input_list_translated, 'output': output_list_translated})

    # Print the original and translated DataFrames for comparison
    print("Original subset:")
    print(subset_df)
    print("\nTranslated subset:")
    print(translated_df)

    # Save the translated subset to a JSON file
    translated_dict = translated_df.to_dict('records')
    write_json_file(translated_dict, f'{output_dir}subset_translated.json')

# Call the function with the desired subset size
translate_dataframe_subset(df, 5)


# Combine translated JSON files into a single JSON file
def combine_chunks():
    translated_tasks_list = []
    for index in range(0, len(glob.glob(f'{output_dir}*.json'))):
        with open(f'{output_dir}chunk{index}.json', "rb") as f:
            translated_tasks_list += json.loads(f.read())
    write_json_file(translated_tasks_list, f'./translated_tasks_vi.json')

combine_chunks()
