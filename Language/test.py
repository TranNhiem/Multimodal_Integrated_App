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
import torch
import string
import requests
import time 
import random
import math 
from ratelimit import limits, sleep_and_retry

from concurrent.futures import ThreadPoolExecutor
import concurrent
## Preprocessing Text
import re
import backoff 
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# import nltk

##****************************************************************
### Section 1 Translation Using Paid API 
##****************************************************************

API_TYPE = "azure"
API_BASE = "https://sslgroupservice.openai.azure.com/"
API_VERSION = "2023-03-15-preview" #"2022-06-01-preview"#"2023-03-15-preview"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-35-turbo"#"gpt-3.5-turbo" #"gpt-35-turbo" for Azure API, OpenAI API "gpt-3.5-turbo"#"gpt-4", "text-davinci-003"

TARGET_LANGUAGE = "Vietnamese language" #"Vietnamese language"
CHUNK_SIZE = 5
OUTPUT_DIR = "./data/output/"
import random
# Set up API
def setup_api(api="azure"):
    if api == "azure":
        openai.api_type = API_TYPE
        openai.api_base = API_BASE
        openai.api_version = API_VERSION
        openai.api_key = API_KEY
    else:
        openai.organization = "org-PVVobcsgsTm9RT8Ez5DubzbX" # Central IT account
        #openai.api_key = API_KEY
        openai.api_key = os.getenv("OPENAI_API_KEY")

# Load input data as DataFrame
def load_input_data(INPUT_TASKS_PATH):
    with open(INPUT_TASKS_PATH, "rb") as f:
        json_data = json.loads(f.read())
    return pd.DataFrame(json_data)

# Save the given data to a JSON file at the specified file path
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

##----------- Start PREPROCESSING TEXT ------------------------

import spacy

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

def remove_urls(text):
    # Implementation using SpaCy
    doc = nlp(text)
    text_without_urls = " ".join([token.text for token in doc if not token.like_url])
    return text_without_urls

def remove_html_tags(text):
    # Implementation using regular expressions
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def matches_regex(regex, text):
    return bool(re.compile(regex).search(text))
def remove_special_characters(text, keep_chars="'.,!?"):
    pattern = re.compile(f'[^A-Za-z0-9{keep_chars}\s]')
    return pattern.sub(r'', text)

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

def preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False, stemming=False, lemmatization=False, keep_chars="'.,!?", remove_code=False):
    
    def remove_punctuation(text):
        return ''.join(c if c not in string.punctuation or c == '-' else ' ' for c in text)

    # Remove URLs using SpaCy
    text = remove_urls(text)

    # Remove HTML tags
    text = remove_html_tags(text)

    # Remove special characters
    text = remove_special_characters(text, keep_chars=keep_chars)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove code content
    if remove_code:
        text = re.sub(r'(?s)(?P<tag><code>.*?</code>)', '', text)

    if remove_digits:
        text = re.sub(r'\d+', '', text)

    if to_lowercase:
        text = text.lower()

    # Call the remove_punctuation function
    text = remove_punctuation(text)

    if remove_stopwords or stemming or lemmatization:
        # Tokenize the text using SpaCy
        doc = nlp(text)

        if remove_stopwords:
            # Remove stop words using SpaCy
            tokens = [token.text for token in doc if not token.is_stop]
        else:
            tokens = [token.text for token in doc]

        if stemming:
            # Perform stemming using SpaCy's Lemmatizer
            tokens = [token.lemma_ for token in doc]

        if lemmatization:
            # Perform lemmatization using SpaCy's Lemmatizer
            tokens = [token.lemma_ for token in doc]

        text = ' '.join(tokens)

    return text

# Check if the given text contains words
def contains_words(text):
    return matches_regex(r'[A-z]{3,}', text)

# Check if the given text is translatable
def is_translatable(text):
    if text == "":
        return False
    return (contains_code(text) is False) and contains_words(text)

##-----------END  PREPROCESSING TEXT ------------------------
# Delay between API calls to stay within rate limits
def delay_between_requests():
    time.sleep(7)  # Adjust the delay time as needed


# Define a retry decorator with exponential backoff
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 10,
    exponential_base: float = 20,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.error.RateLimitError,),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper

# Define the rate limit (requests per minute) and token limit
## GPT3
# RATE_LIMIT = 120  # Adjust the rate limit as per your model
# TOKEN_LIMIT = 40000  # Adjust the token limit as per your model
## ChatGPT
RATE_LIMIT = 120
TOKEN_LIMIT = 120000

# ##GPT4
# RATE_LIMIT = 18
# TOKEN_LIMIT = 600000

# Decorator to enforce rate limit
# @sleep_and_retry
# @limits(calls=RATE_LIMIT, period=60)
#@retry_with_exponential_backoff
@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.InvalidRequestError), max_tries=10, max_time=60)
def translate_text_openai(text):
    #delay_between_requests()  # Add delay before each API call
    if not text.strip():
        return ""
    # if ' ' in text:
    #     prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    # else:
    #     prompt= f'Please provide the {TARGET_LANGUAGE} translation for the following word: {text}'
    #prompt = f"Please translate the following English text to {TARGET_LANGUAGE} : {text}"
    # prompt= f" English text: {text} translation into Traditional Chinese language: " # Not greate result
    # prompt= f"Translate the following English text to Traditional language: {text}"
    # prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    # prompt = f'Translate the following English text into {TARGET_LANGUAGE}: "{text}"'
    
    # response = openai.Completion.create(
    #     engine=MODEL, 
    #     prompt=prompt, 
    #     max_tokens=800, 
    #     stop=None, 
    #     temperature=0.01,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )
    # translated_text = response.choices[0].text.strip()
    # return translated_text.split('\n')[-1].strip()
    response = openai.ChatCompletion.create(
    engine=MODEL,
    messages=[
        {"role": "system", "content": f'Translate the following English text into {TARGET_LANGUAGE}:'},
        {"role": "user", "content": text}
    ],
    max_tokens=400,
    temperature=0.3,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0)
    translated_text = response.choices[0].message.content.strip()
    
    return translated_text


def translate_text_openai_with_backoff(text):
    retries = 0
    max_retries = 5
    base_delay = 20  # Initial delay in seconds
    max_delay = 70  # Maximum delay in seconds

    while retries < max_retries:
        try:
            return translate_text_openai(text)
        except openai.error.RateLimitError:
            delay = random.uniform(0, min(max_delay, base_delay * math.pow(2, retries)))
            time.sleep(delay)
            retries += 1
        except openai.error.Timeout as e:
            # Handle timeout error
            print(f"Request timed out: {e}")
            delay = random.uniform(0, min(max_delay, base_delay * math.pow(2, retries)))
            time.sleep(delay)
            retries += 1

    # If all retries are exhausted, raise an error
    raise Exception("Rate limit exceeded. Retry limit reached.")

## Save the translated subset to a JSON file
def save_translated_subset_to_json(translated_subset_df, file_path):
    translated_subset_dict = translated_subset_df.to_dict('records')
    # with open(file_path, 'w') as outfile:
    #     json.dump(translated_subset_dict, outfile)
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(translated_subset_dict, outfile, ensure_ascii=False)
     # Translate a single text string

#@retry_with_exponential_backoff
def process_chunks_openai(chunks):
    translated_texts = []
    for text in chunks:
        if is_translatable(text):
            preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=True, remove_code=True)
            translated_text = translate_text_openai(preprocessed_text)
            #translated_text= translate_text_openai_with_backoff(preprocessed_text)
            translated_texts.append(translated_text)
        else:
            translated_texts.append(text)
    return translated_texts

def translate_text_openai_parallel(texts, chunk_size=50):
    chunked_texts = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    # print(f'Chunks Text before translate: {len(chunked_texts)}')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunks_openai, chunk) for chunk in chunked_texts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    translated_texts = []
    for result in results:
        translated_texts.extend(result)

    return translated_texts

def translate_text_with_error_handling(text):
    try:
        return translate_text_openai(text)
    except openai.error.InvalidRequestError:
        print(f"Skipping data due to content filtering: {text}")
        return None

# Update the test_translation function to use translate_text_openai_parallel
def test_translation_update(df, start=0, end=4, subset=True):
    if subset:
        subset_df = df.iloc[start:end]
    else:
        subset_df = df

    # # translated_instructions = []
    # # translated_inputs = []
    # # translated_outputs = []
   
   
    # try:
    #     translated_instructions = translate_text_openai_parallel(subset_df['instruction'].tolist())
    #     translated_inputs = translate_text_openai_parallel(subset_df['input'].tolist())
    #     translated_outputs = translate_text_openai_parallel(subset_df['output'].tolist())
    # # translated_instructions.extend(translate_text_openai_parallel(subset_df['instruction'].tolist()))
    # # translated_inputs.extend(translate_text_openai_parallel(subset_df['input'].tolist()))
    # # translated_outputs.extend(translate_text_openai_parallel(subset_df['output'].tolist()))
    # except openai.error.InvalidRequestError:
    #     print("Skipping data due to content filtering.")
    
    # translated_subset_df = pd.DataFrame({
    # 'instruction': translated_instructions,
    # 'input': translated_inputs,
    # 'output': translated_outputs
    # })

    # save_translated_subset_to_json(
    #     translated_subset_df,
    #     f'./data/output/Vietnamese_Translation_Azure_GPT_35_{start}_{end}_new.json'
    # )


    
    # except Exception as e:
    #     ##Raise the exception again to halt the program
    #     raise e

    translated_instructions = []
    translated_inputs = []
    translated_outputs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _, row in subset_df.iterrows():
            instruction = row['instruction']
            input_text = row['input']
            output_text = row['output']
            futures.append(executor.submit(translate_text_with_error_handling, instruction))
            futures.append(executor.submit(translate_text_with_error_handling, input_text))
            futures.append(executor.submit(translate_text_with_error_handling, output_text))

        for future in concurrent.futures.as_completed(futures):
            translated_text = future.result()
            if translated_text is not None:
                if len(translated_instructions) < len(subset_df):
                    translated_instructions.append(translated_text)
                elif len(translated_inputs) < len(subset_df):
                    translated_inputs.append(translated_text)
                elif len(translated_outputs) < len(subset_df):
                    translated_outputs.append(translated_text)

    translated_subset_df = pd.DataFrame({
        'instruction': translated_instructions,
        'input': translated_inputs,
        'output': translated_outputs
    })

    save_translated_subset_to_json(
        translated_subset_df,
        f'./data/output/Vietnamese_Translation_Azure_GPT_35_{start}_{end}.json'
    )

    #save_translated_subset_to_json(translated_subset_df, './data/output/Vietnamese_Translation_Azure_GPT_35_0_10K.json')

def main():
        setup_api(api="azure") # "azure"
        input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
        ## get the length of the dataframe
        # df_length = len(input_data)
        # # print the length
        ## Old Version 
        #test_translation(input_data, start=0,end=10000, subset=True)
        #print(f"The length of the dataframe is: {df_length}")
        # test_translation_update(input_data, start=0,end=10000, subset=True)
        ##--------------------Another Tried via Saving Automatically --------------------------
        input_data= input_data.iloc[0:20]
        # start = 0 ## Change this start
        # end = 10000 ## 10000
        # while start < end:
        #     try:
        #         test_translation_update(input_data, start=start, end=end, subset=True)
        #     except Exception as e:
        #         print(f"Error occurred: {str(e)}")
        #         # Update the start position to resume from the next subset
        #         start = end
        #         # Update the end position for the next subset
        #         end = min(end + 1000, len(input_data))
        #     else:
        #         # If no exception occurred, update the start and end positions for the next subset
        #         start = end
        #         end = min(end + 1000, len(input_data))


        # Define the subset size (e.g. 1000)
        subset_size = 2
        
        # Initialize start position
        start = 0
        end=10
        test_translation_update(input_data[start:end], start=start, end=end, subset=True)
        
        # while start < len(input_data):
        #     end = min(start + subset_size, len(input_data))
        #     try:
        #         test_translation_update(input_data[start:end], start=start, end=end, subset=True)
        #     except Exception as e:
        #         print(f"Error occurred: {str(e)}")
        #         # Update the start position to resume from the next subset
        #         start = end
        #         # Update the end position for the next subset
        #         end = min(end + subset_size, len(input_data))
        #     else:
        #         # If no exception occurred, update the start and end positions for the next subset
        #         start = end
        
        ##--------------------Another Tried via Delay time --------------------------

        # # Split input data into smaller batches for translation
        # start=10000
        # end=20000
        # input_data = input_data.iloc[start:end]
        # # Split input data into smaller batches for translation
        # batch_size = 10
        # num_batches = (len(input_data) // batch_size) + 1
        # translated_instructions = []
        # translated_inputs = []
        # translated_outputs = []
        # for i in range(num_batches):
        #     start_idx = i * batch_size
        #     end_idx = (i + 1) * batch_size
        #     batch_instruction = input_data.iloc[start_idx:end_idx]['instruction'].tolist()
        #     batch_input = input_data.iloc[start_idx:end_idx]['input'].tolist()
        #     batch_output = input_data.iloc[start_idx:end_idx]['output'].tolist()

        #     # Translate batches of instructions, inputs, and outputs in parallel
        #     translated_batch_instructions = translate_text_openai_parallel(batch_instruction)
        #     translated_batch_inputs = translate_text_openai_parallel(batch_input)
        #     translated_batch_outputs = translate_text_openai_parallel(batch_output)

        #     translated_instructions.extend(translated_batch_instructions)
        #     translated_inputs.extend(translated_batch_inputs)
        #     translated_outputs.extend(translated_batch_outputs)

        #     # Add delay between batches to stay within rate limits
        #     time.sleep(4)

        #     # Create a new DataFrame with translated data
        # translated_df = pd.DataFrame({
        #     'instruction': translated_instructions,
        #     'input': translated_inputs,
        #     'output': translated_outputs
        # })

        # # Save the translated subset to a JSON file
        # save_translated_subset_to_json(translated_df, "'./data/output/Vietnamese_Translation_Azure_GPT_35_10_20K.json'") 

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")