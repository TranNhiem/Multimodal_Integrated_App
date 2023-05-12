'''
@TranNhiem 2023/05
This design including 2 Sections:

1. Using The Pay API to Translate Dataset
    + OpenAI API (gpt-3.5-turbo) & GPT-3 API (text-davinci-003)
    + Azure Translate Optional 
    + Google Translation API Optional 
2. Using Open-Source Pretrained Language Model for Transslation 
    + NLLB - MetaAI Translation Model 
    + BLOOM - Opensource Multi-lingual Language Model
    + T5&FlanT5 - Google's Text-to-Text 
'''
import os
import openai
import json
import pandas as pd
import numpy as np
import re
import glob


import requests


##  Download Original English Dataset Version
#url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
# url="https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
# response = requests.get(url)

# output_file = "./data/alpaca_52k_instruction_cleaned.json"
# if response.status_code == 200:
#     with open(output_file, 'wb') as f:
#         f.write(response.content)
#     print(f"File downloaded successfully and saved as {output_file}")
# else:
#     print(f"Failed to download the file. Status code: {response.status_code}")
# Constants


##****************************************************************
### Section 1 Translation Using Paid API 
##****************************************************************

API_TYPE = "azure"
API_BASE = "https://sslgroupservice.openai.azure.com/"
API_VERSION = "2023-03-15-preview" #"2022-06-01-preview"#"2023-03-15-preview"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-davinci-003"#"gpt-3.5-turbo" #"gpt-35-turbo" for Azure API, OpenAI API "gpt-3.5-turbo"#"gpt-4", "text-davinci-003"


TARGET_LANGUAGE = "Tranditional Chinese language" #"Vietnamese language"
CHUNK_SIZE = 5
OUTPUT_DIR = "./data/output/"

# Set up API
def setup_api(api="azure"):
    if api == "azure":
        openai.api_type = API_TYPE
        openai.api_base = API_BASE
        openai.api_version = API_VERSION
        openai.api_key = API_KEY
    else:
        openai.organization = "org-PVVobcsgsTm9RT8Ez5DubzbX" # Central IT account
        openai.api_key = API_KEY

# Load input data as DataFrame
def load_input_data(INPUT_TASKS_PATH):
    with open(INPUT_TASKS_PATH, "rb") as f:
        json_data = json.loads(f.read())
    return pd.DataFrame(json_data)

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

def translate_text_openai(text):
    if not text.strip():
        return ""
    if ' ' in text:
        prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    else:
        prompt= f'Please provide the {TARGET_LANGUAGE} translation for the following word: {text}'
    #prompt = f"Please translate the following English text to {TARGET_LANGUAGE} : {text}"
    # prompt= f" English text: {text} translation into Vietnamese language: " # Not greate result
    # prompt=f"Translate the following English text to Vietnamese language: {text}"
    # prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    response = openai.Completion.create(
        engine=MODEL, 
        prompt=prompt, 
        max_tokens=800, 
        stop=None, 
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    translated_text = response.choices[0].text.strip()
    return translated_text.split('\n')[-1].strip()

# Translate a DataFrame
def translate_dataframe(df):
    translated_df = df.copy()
    for column in df.columns:
        translated_df[column] = df[column].apply(lambda x: translate_text_openai(x) if is_translatable(x) else x)
    return translated_df

# Save translated DataFrame to JSON
def save_translated_dataframe(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    translated_data = df.to_dict('records')
    output_file = f"{OUTPUT_DIR}translated_tasks_{TARGET_LANGUAGE}.json"
    write_json_file(translated_data, output_file)

## Save the translated subset to a JSON file
def save_translated_subset_to_json(translated_subset_df, file_path):
    translated_subset_dict = translated_subset_df.to_dict('records')
    # with open(file_path, 'w') as outfile:
    #     json.dump(translated_subset_dict, outfile)
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(translated_subset_dict, outfile, ensure_ascii=False)
        
# Translate a subset of the DataFrame
def translate_dataframe_subset(series, n_rows, subset=True):
    if subset:
        subset_series = series.head(n_rows)
    else: 
        subset_series = series

    # Apply translation to the subset series
    translated_series = subset_series.apply(lambda x: translate_text_openai(x) if is_translatable(x) else x)

    # Print out the translations For Debugging
    # for index, value in subset_series.items():
    #     if is_translatable(value):
    #         print(f"Before: {value}\nAfter: {translate_text_openai(value)}\n")
    return translated_series

# def translate_dataframe_subset(instruction_series,  n_rows, subset=True, output_file):
#     translated_instructions = []
#     if subset:
#         subset_series = instruction_series.head(n_rows)
#     else: 
#         instruction_series = instruction_series

#     # Iterate through all rows in the instruction_series
#     for index, value in instruction_series.items():
#         translated_text = translate_text_openai(value)
#         translated_instructions.append(translated_text)

#         # Save the progress after each translation
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(translated_instructions, f, ensure_ascii=False, indent=4)
#     return translated_instructions


## Save the translated subset to a JSON file
def test_translation(df, n_rows=4, subset=True):
    
    if subset:
        subset_df = df.head(n_rows)
        translated_instruction = translate_dataframe_subset(subset_df['instruction'], n_rows, subset=subset)
        translated_input = translate_dataframe_subset(subset_df['input'], n_rows,  subset=subset)
        translated_output = translate_dataframe_subset(subset_df['output'], n_rows,  subset=subset)
    
    else: 
        translated_instruction = translate_dataframe_subset(df['instruction'], n_rows,  subset=subset)
        translated_input = translate_dataframe_subset(df['input'], n_rows,  subset=subset)
        translated_output = translate_dataframe_subset(df['output'], n_rows,  subset=subset)
    
  
    
    translated_subset_df = pd.DataFrame({'instruction': translated_instruction, 
                                         'input': translated_input, 
                                         'output': translated_output})
    
    save_translated_subset_to_json(translated_subset_df, './data/output/translated_Traditional_Chinese_ChatGPT_OpenAI_2023_newprompt3.json')
    
    # print("\nOriginal subset:")
    # print(subset_df)
    # print("\nTranslated subset:")
    # print(translated_subset_df)


def main():
    setup_api(api="openAI") # "azure"
    input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
    ## get the length of the dataframe
    # df_length = len(input_data)
    # # print the length
    # print(f"The length of the dataframe is: {df_length}")
    test_translation(input_data, n_rows=10, subset=True)

##****************************************************************
### Section 2 Translation Using Open-Source Pretrained Model
##****************************************************************
'''
'''



if __name__ == "__main__":
    main()
