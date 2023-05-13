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

from concurrent.futures import ThreadPoolExecutor
import concurrent
## Preprocessing Text
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
#nltk.data.path.append('/home/rick/nltk_data')
# nltk.download('stopwords')  # Add this line to download stopwords explicitly
# nltk.download('wordnet')
# nltk.data.path.append('/path/to/nltk_data')

# nltk.download()
#nltk.download("wordnet", force=True)

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

TARGET_LANGUAGE = "Traditional Chinese language" #"Vietnamese language"
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

##----------- Start PREPROCESSING TEXT ------------------------

# def remove_urls(text):
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def remove_html_tags(text):
#     html_pattern = re.compile(r'<.*?>')
#     return html_pattern.sub(r'', text)

# def remove_special_characters(text, keep_chars="'.,!?"):
#     pattern = re.compile(f'[^A-Za-z0-9{keep_chars}\s]')
#     return pattern.sub(r'', text)

# def matches_regex(regex, text):
#     return bool(re.compile(regex).search(text))

# def contains_code(text):
#     code_blacklist = ['&&', '||', '<html>', ';\n', 'SELECT']
#     return (
#         any(code_keyword in text for code_keyword in code_blacklist) or
#         matches_regex(r'\w+\(\w*\) \{', text) or
#         matches_regex(r'def \w+\(', text) or
#         matches_regex(r'\[A-z]+\.[A-z]+', text) or
#         matches_regex(r': [\w\.#]{1,12};', text) or
#         matches_regex(r'<\/\w+>', text)
#     )

# def preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False, stemming=False, lemmatization=False, keep_chars="'.,!?", remove_code=False):
    
#     def remove_punctuation(text):
#         return ''.join(c if c not in string.punctuation or c == '-' else ' ' for c in text)
  
#     # Remove URLs
#     text = remove_urls(text)

#     # Remove HTML tags
#     text = remove_html_tags(text)

#     # Remove special characters
#     text = remove_special_characters(text, keep_chars=keep_chars)

#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()

#     # Remove code content
#     if remove_code:
#         text = re.sub(r'(?s)(?P<tag><code>.*?</code>)', '', text)

#     if remove_digits:
#         text = re.sub(r'\d+', '', text)

#     if to_lowercase:
#         text = text.lower()
#     # Call the remove_punctuation function
#     text = remove_punctuation(text)
   
    
#     if remove_stopwords or stemming or lemmatization:
#         tokens = word_tokenize(text)
#         if remove_stopwords:
#             #stop_words = set(stopwords.words('english'))
#             stop_words = set(stopwords.words('english')).union(set(stopwords.words('english')))
#             text = " ".join([word for word in text.split() if word not in stop_words])
#         if stemming:
#             stemmer = PorterStemmer()
#             tokens = [stemmer.stem(token) for token in tokens]

#         if lemmatization:
#             lemmatizer = WordNetLemmatizer()
#             tokens = [lemmatizer.lemmatize(token) for token in tokens]

#         text = ' '.join(tokens)

#     return text

# # Check if the given text contains words
# def contains_words(text):
#     return matches_regex(r'[A-z]{3,}', text)

# # Check if the given text is translatable
# def is_translatable(text):
#     if text == "":
#         return False
#     return (contains_code(text) is False) and contains_words(text)

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

def translate_text_openai(text):
    if not text.strip():
        return ""
    # if ' ' in text:
    #     prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    # else:
    #     prompt= f'Please provide the {TARGET_LANGUAGE} translation for the following word: {text}'
    #prompt = f"Please translate the following English text to {TARGET_LANGUAGE} : {text}"
    # prompt= f" English text: {text} translation into Traditional Chinese language: " # Not greate result
    # prompt=f"Translate the following English text to Tradutuib language: {text}"
    # prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    prompt = f'Translate the following English text into {TARGET_LANGUAGE}: "{text}"'
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

## Save the translated subset to a JSON file
def save_translated_subset_to_json(translated_subset_df, file_path):
    translated_subset_dict = translated_subset_df.to_dict('records')
    # with open(file_path, 'w') as outfile:
    #     json.dump(translated_subset_dict, outfile)
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(translated_subset_dict, outfile, ensure_ascii=False)
     # Translate a single text string

def translate_text(text):
    if is_translatable(text):
        preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=False, remove_code=True)
        translated_text = translate_text_openai(preprocessed_text)
        return translated_text
    else:
        return text

# Save the translated subset to a JSON file
def test_translation(df, start=0,end=4, subset=True):
    if subset:
        #subset_df = df.head(n_rows)
        subset_df= df.iloc[start:end]
    else:
        subset_df = df

    translated_instruction = subset_df['instruction'].apply(translate_text)
    translated_input = subset_df['input'].apply(translate_text)
    translated_output = subset_df['output'].apply(translate_text)

    translated_subset_df = pd.DataFrame({'instruction': translated_instruction, 
                                         'input': translated_input, 
                                         'output': translated_output})
    
    save_translated_subset_to_json(translated_subset_df, './data/output/translated_Traditional_Chinese_GPT3_2023_newprompt3.json')

    # print("\nOriginal subset:")
    # print(subset_df)
    # print("\nTranslated subset:")
    # print(translated_subset_df)

### Update The Code To Process Text into Chunk & also Using Multi-Thread 
def process_chunks_openai(chunks):
    translated_texts = []
    for text in chunks:
        if is_translatable(text):
            preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=True, remove_code=True)
            translated_text = translate_text_openai(preprocessed_text)
            translated_texts.append(translated_text)
        else:
            translated_texts.append(text)
    return translated_texts


def translate_text_openai_parallel(texts, chunk_size=10):
    chunked_texts = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunks_openai, chunk) for chunk in chunked_texts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    translated_texts = []
    for result in results:
        translated_texts.extend(result)

    return translated_texts

# Update the test_translation function to use translate_text_openai_parallel
def test_translation_update(df, start=0, end=4, subset=True):
    if subset:
        subset_df = df.iloc[start:end]
    else:
        subset_df = df

    translated_instructions = translate_text_openai_parallel(subset_df['instruction'].tolist())
    translated_inputs = translate_text_openai_parallel(subset_df['input'].tolist())
    translated_outputs = translate_text_openai_parallel(subset_df['output'].tolist())

    translated_subset_df = pd.DataFrame({'instruction': translated_instructions,
                                         'input': translated_inputs,
                                         'output': translated_outputs})

    save_translated_subset_to_json(translated_subset_df, './data/output/translated_Traditional_Chinese_GPT3_0k_10k.json')

# def main():
#         setup_api(api="azure") # "azure"
#         input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
#         ## get the length of the dataframe
#         # df_length = len(input_data)
#         # # print the length
#         ## Old Version 
#         #test_translation(input_data, start=0,end=10000, subset=True)
#         # print(f"The length of the dataframe is: {df_length}")
#         test_translation_update(input_data, start=0,end=10000, subset=True)


##****************************************************************
### Section 2 Translation Using Open-Source Pretrained Model
##****************************************************************

## Libary for Language Translation Model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, NllbTokenizer
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Using NLLB open-source Model 
source_langage={
    "🇱🇷 English": "eng_Latn",
    "🇻🇳 Vietnamese": "vie_Latn", 
    "TraditionalChinese": "zho_Hant",
    "🇨🇳 SimplifiedChinese": "zho_Hans",
    "🇫🇷 French" : "fra_Latn",
    "🇩🇪 German": "deu_Latn",
    "🇲🇨 Indonesian": "ind_Latn",
    "🇯🇵 Japanese": "jpn_Jpan",
    "🇰🇷 Korean": "kor_Hang", 
    "🇪🇸 Spanish": "spa_Latn", 
    "🇹🇭 Thai": "tha_Thai",
    "": "empty",
}
## Language Translation
#samples_num=int(samples_num/2)
weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/NLLB/"
# Create the weight_path if it is not exist 
if not os.path.exists(weight_path): 
    os.makedirs(weight_path)
# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Translation function
async def translate_text_nllb(text, source_language, target_language):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=600)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
        max_length=800,
        early_stopping=True
    )

    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

# def translate_text_nllb(text, source_language, target_language):
#     tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B",cache_dir=weight_path )#cache_dir=NLLB_path
    
#     #tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
   
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model=model.to(device)
#     # # Check if GPU is available
#     # Tokenize and encode the text
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=600)
    
#     # Move the input tokens to the GPU if available
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     # translator_prompt = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_language, tgt_lang=target_language, max_length=800)
#     # prompt = translator_prompt(text)[0]
#     # translated_text = prompt['translation_text']

#     # Generate the translation
#     translated_tokens = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
#         max_length=800,
#         early_stopping=True
#     )

#     # Decode the translation
#     translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

#     # del translator_prompt
#     # del tokenizer
#     # del model
#     #torch.cuda.empty_cache()
#     return translated_text

def translate_dataframe_subset(text):
    if is_translatable(text):
        preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=True, remove_code=True)
        translated_text = translate_text_nllb(preprocessed_text, source_langage['🇱🇷 English'], source_langage['🇻🇳 Vietnamese'])
        return translated_text
    else:
        return text

def translate_and_save_to_json(dataset, output_file, start, end, subset=True):
    translations = []

    if subset:
        #dataset = dataset.head(end)
        dataset= dataset.iloc[start:end]

    for index, data in dataset.iterrows():
        instruction = data["instruction"]
        input_text = data["input"]
        output_text = data["output"]

        translated_instruction = translate_dataframe_subset(instruction)
        translated_input = translate_dataframe_subset(input_text)
        translated_output = translate_dataframe_subset(output_text)

        translations.append({
            "instruction": translated_instruction,
            "input": translated_input,
            "output": translated_output
        })

    save_translated_subset_to_json(pd.DataFrame(translations), output_file)



### Processing Text into Chunk of Data 
def process_chunks_nllb(chunks):
    translated_texts = []
    for text in chunks:
        if is_translatable(text):
            preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=True, remove_code=True)
            translated_text = translate_text_nllb(preprocessed_text, source_langage['🇱🇷 English'], source_langage['🇻🇳 Vietnamese'])
            translated_texts.append(translated_text)
        else:
            translated_texts.append(text)
    return translated_texts

def translate_text_nllb_parallel(texts, chunk_size=70):
    
    chunked_texts = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunks_nllb, chunk) for chunk in chunked_texts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    translated_texts = []
    for result in results:
        translated_texts.extend(result)

    return translated_texts

def translate_and_save_to_json_update(dataset, output_file, start, end, subset=True):
    translations = []
    if subset:
        dataset = dataset.iloc[start:end]

    translated_instructions = translate_text_nllb_parallel(dataset['instruction'].tolist())
    translated_inputs = translate_text_nllb_parallel(dataset['input'].tolist())
    translated_outputs = translate_text_nllb_parallel(dataset['output'].tolist())

    for instruction, input_text, output_text in zip(translated_instructions, translated_inputs, translated_outputs):
        translations.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    save_translated_subset_to_json(pd.DataFrame(translations), output_file)

# def main():
#     input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
#     df_length = len(input_data)
#     # # print the length
#     print(f"The length of the dataframe is: {df_length}")
#     translate_and_save_to_json_update(input_data, output_file="./data/output/NLLB_translations_Vietnamese_test.json", start=0, end=4,subset=True)

# Main function
async def main():
    input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
    df_length = len(input_data)
    print(f"The length of the dataframe is: {df_length}")

    # Translate in parallel
    translated_texts = await asyncio.gather(
        *[translate_text_nllb(preprocess_text(text), source_language['🇱🇷 English'], source_language['🇻🇳 Vietnamese']) for text in input_data['instruction']]
    )

    # Store the translations in a DataFrame
    translations_df = pd.DataFrame({
        "instruction": translated_texts,
        "input": input_data["input"],
        "output": input_data["output"]
    })

    # Save the translations to a JSON file
    save_translated_subset_to_json(translations_df, output_file="./data/output/NLLB_translations_Vietnamese_test_1.json")

# Run the asyncio event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(main())


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")