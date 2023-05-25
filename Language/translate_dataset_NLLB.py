'''
TranNhiem 2023/05/12 

I made the following improvements to Run inference LLM:

1. Converted the main function to an asynchronous function (async def main(start, end, subset)).
2. Used the asyncio.gather function to run the translate_text_nllb function in parallel for all three columns (instruction, input, output) of the input data.
3. Modified the translation process to preprocess each text before translating it using the preprocess_text function.
4. Created separate lists for translated instructions, inputs, and outputs.
5. Stored the translations in a DataFrame (translations_df) with the appropriate column names.
6. Saved the translations to a JSON file using the save_translated_subset_to_json function.

'''


import re
import string
import pandas as pd
import concurrent.futures
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time 
import os 
import spacy
import torch
import json
# Load the SpaCy English language model
# nlp = spacy.load("en_core_web_sm")

def remove_urls(text):
    """
    Remove URLs from the text using SpaCy.

    Args:
        text (str): Input text.

    Returns:
        str: Text with URLs removed.
    """
    doc = nlp(text)
    text_without_urls = " ".join([token.text for token in doc if not token.like_url])
    return text_without_urls

def remove_html_tags(text):
    """
    Remove HTML tags from the text using regular expressions.

    Args:
        text (str): Input text.

    Returns:
        str: Text with HTML tags removed.
    """
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def matches_regex(regex, text):
    """
    Check if the text matches the given regex pattern.

    Args:
        regex (str): Regular expression pattern.
        text (str): Input text.

    Returns:
        bool: True if the text matches the pattern, False otherwise.
    """
    return bool(re.compile(regex).search(text))

def remove_special_characters(text, keep_chars="'.,!?"):
    """
    Remove special characters from the text, except for the specified keep_chars.

    Args:
        text (str): Input text.
        keep_chars (str): Characters to keep in the text.

    Returns:
        str: Text with special characters removed.
    """
    pattern = re.compile(f'[^A-Za-z0-9{keep_chars}\s]')
    return pattern.sub(r'', text)

def contains_code(text):
    """
    Check if the text contains code snippets.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text contains code, False otherwise.
    """
    code_blacklist = ['&&', '||', '<html>', ';\n', 'SELECT']
    return (
        any(code_keyword in text for code_keyword in code_blacklist) or
        matches_regex(r'\w+\(\w*\) \{', text) or
        matches_regex(r'def \w+\(', text) or
        matches_regex(r'\[A-z]+\.[A-z]+', text) or
        matches_regex(r': [\w\.#]{1,12};', text) or
        matches_regex(r'<\/\w+>', text)
    )

def preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False,
                    stemming=False, lemmatization=False, keep_chars="'.,!?", remove_code=False):
    """
    Preprocess the text by removing URLs, HTML tags, special characters, digits, punctuation,
    and applying lowercase, stopword removal, stemming, or lemmatization if specified.

    Args:
        text (str): Input text.
        remove_digits (bool): Whether to remove digits from the text.
        to_lowercase (bool): Whether to convert the text to lowercase.
        remove_stopwords (bool): Whether to remove stopwords using SpaCy.
        stemming (bool): Whether to perform stemming using SpaCy's Lemmatizer.
        lemmatization (bool): Whether to perform lemmatization using SpaCy's Lemmatizer.
        keep_chars (str): Characters to keep in the text.
        remove_code (bool): Whether to remove code snippets from the text.

    Returns:
    str: Processed text.
    """
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

def contains_words(text):
    """
    Check if the text contains words.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text contains words, False otherwise.
    """
    return matches_regex(r'[A-z]{3,}', text)

def is_translatable(text):
    """
    Check if the given text is translatable.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text is translatable, False otherwise.
    """
    if text == "":
        return False
    return (contains_code(text) is False) and contains_words(text)

def load_input_data(INPUT_TASKS_PATH):
    """
    Load input data from a JSON file and return as a DataFrame.

    Args:
        INPUT_TASKS_PATH (str): Path to the input JSON file.

    Returns:
        pd.DataFrame: Input data as a DataFrame.
    """
    with open(INPUT_TASKS_PATH, "rb") as f:
        json_data = json.loads(f.read())
    return pd.DataFrame(json_data)

## Save the translated subset to a JSON file
def save_translated_subset_to_json(translated_subset_df, file_path):
    """
    Save the translated subset DataFrame to a JSON file.

    Args:
        translated_subset_df (pd.DataFrame): Translated subset as a DataFrame.
        file_path (str): Output file path.
    """
    translated_subset_dict = translated_subset_df.to_dict('records')
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(translated_subset_dict, outfile, ensure_ascii=False)

# weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/NLLB/"
# #Create the weight_path if it is not exist 
# if not os.path.exists(weight_path): 
#     os.makedirs(weight_path)
# #Initialize the tokenizer and model
# #Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Initialize the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B",cache_dir=weight_path)
# model = model.to(device)
source_language={
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
# Translation function
# async def translate_text_nllb(text, source_language, target_language):
#     """
#     Translate the text using the NLLB model.

#     Args:
#         text (str): Input text.
#         source_language (str): Source language code.
#         target_language (str): Target language code.

#     Returns:
#         str: Translated text.
#     """
#     # Checking whether the text is translatable or not
#     if not is_translatable(text):
#         return text
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=600)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     translated_tokens = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
#         max_length=800,
#         early_stopping=True
#     )

#     translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
#     return translated_text

# # Main function
# async def main(start, end, subset):
#     """
#     Main function to perform translation.

#     Args:
#         start (int): Start index of the data subset.
#         end (int): End index of the data subset.
#         subset (bool): Whether to use a subset of the data.

#     Returns:
#         None
#     """
#     input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")

#     # Subset the data if needed
#     if subset:
#         input_data = input_data.iloc[start:end]

#     df_length = len(input_data)
#     print(f"The length of the dataframe is: {df_length}")

 
#     # Translate in parallel
#     translations = await asyncio.gather(
#         *[translate_text_nllb(preprocess_text(text), source_language['🇱🇷 English'], source_language['TraditionalChinese']) for text in input_data['instruction']],
#         *[translate_text_nllb(preprocess_text(text), source_language['🇱🇷 English'], source_language['TraditionalChinese']) for text in input_data['input']],
#         *[translate_text_nllb(preprocess_text(text), source_language['🇱🇷 English'], source_language['TraditionalChinese']) for text in input_data['output']]
#             )

#     # Store the translations in separate lists
#     translated_instructions = translations[:len(input_data)]
#     translated_inputs = translations[len(input_data):2*len(input_data)]
#     translated_outputs = translations[2*len(input_data):]
    
#     # Store the translations in a DataFrame
#     translations_df = pd.DataFrame({
#         "instruction": translated_instructions,
#         "input": translated_inputs,
#         "output": translated_outputs
#     })

#     # Save the translations to a JSON file
#     save_translated_subset_to_json(translations_df, file_path="./data/output/NLLB_translations_TraditionalChinese_40_51k76.json")


#--------------------------------------------------------------
## Inference Model with Batch -- Under Development 
#--------------------------------------------------------------
## Translation function
# async def translate_batch_nllb(texts, source_language, target_language):
    # inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=600)
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    # translated_tokens = model.generate(
    #     **inputs,
    #     forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
    #     max_length=800,
    #     early_stopping=True
    # )

    # translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    # return translated_texts
# # Main function

# async def main(start, end, subset):
#     input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")

#     # Subset the data if needed
#     if subset:
#         input_data = input_data.iloc[start:end]

#     df_length = len(input_data)
#     print(f"The length of the dataframe is: {df_length}")

#     # # Initialize the tokenizer and model
#     # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
#     # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=weight_path)
#     # model = model.to(device)

#     # Translate in parallel
#     batch_size = 8
#     translations = []
#     for i in range(0, len(input_data), batch_size):
#         batch_texts = input_data.iloc[i:i + batch_size]['instruction'].tolist()
#         translated_batch = await translate_batch_nllb(batch_texts, source_language['🇱🇷 English'], source_language['🇻🇳 Vietnamese'], )
#         translations.extend(translated_batch)

#     # Store the translations in separate lists
#     translated_instructions = translations[:df_length]
#     translated_inputs = translations[df_length:2 * df_length]
#     translated_outputs = translations[2 * df_length:3 * df_length]

#     # Store the translations in a DataFrame
#     translations_df = pd.DataFrame({
#         "instruction": translated_instructions,
#         "input": translated_inputs,
#         "output": translated_outputs
#     })

#     # Save the translations to a JSON file
#     save_translated_subset_to_json(translations_df, file_path="./data/output/NLLB_translations_Vietnamese_test_1.json")


###---------------------------------
## Section Using BLOOMZ follow Huamn instruction -- https://huggingface.co/bigscience/bloomz-3b 
###---------------------------------
from transformers import AutoModelForCausalLM
from torch.cuda.amp import autocast
# Initialize the BLOOMZ tokenizer and model
checkpoint = "bigscience/bloomz-3b"

weight_path=  "/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLOOMZ/"
if not os.path.exists(weight_path):
    os.makedirs(weight_path)


# Translation function for BLOOMZ model
async def translate_text_bloomz(text, target_language, tokenizer, model):
    # Within the translate_text_bloomz function:
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
   
    with torch.autocast(device_type='cuda'):
        inputs = tokenizer.encode(f"Translate to {target_language}: {text}", return_tensors="pt", truncation=True, max_length=400).to("cuda")
        outputs = model.generate(inputs,  max_length=400,)
        # inputs = tokenizer.encode(f"Translate to {target_language}: {text}", return_tensors="pt").to("cuda")
        # outputs = model.generate(inputs,  max_length=200,)
        translated_text = tokenizer.decode(outputs[0])
    return translated_text

# Main function for BLOOMZ model
async def main_bloomz(start, end, subset):
    # Load input data
    input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,cache_dir=weight_path)# cache_dir=weight_path
    model = AutoModelForCausalLM.from_pretrained(checkpoint,  cache_dir=weight_path) #cache_dir=weight_path,#torch_dtype="auto", device_map="auto",  gradient_checkpointing=True

    # Subset the data if needed
    if subset:
        input_data = input_data.iloc[start:end]

    df_length = len(input_data)
    print(f"The length of the dataframe is: {df_length}")

    # Translate in parallel
    translations = await asyncio.gather(
        *[translate_text_bloomz(text,  target_language='Vietnamese', tokenizer=tokenizer, model=model) for text in input_data['instruction']],
        *[translate_text_bloomz(text,  target_language='Vietnamese', tokenizer=tokenizer, model=model) for text in input_data['input']],
        *[translate_text_bloomz(text, target_language='Vietnamese',  tokenizer=tokenizer, model=model) for text in input_data['output']]

        # *[translate_text_bloomz(preprocess_text(text),  target_language='Vietnamese') for text in input_data['instruction']],
        # *[translate_text_bloomz(preprocess_text(text),  target_language='Vietnamese') for text in input_data['input']],
        # *[translate_text_bloomz(preprocess_text(text), target_language='Vietnamese') for text in input_data['output']]
    
    )

    # Store the translations in separate lists
    translated_instructions = translations[:len(input_data)]
    translated_inputs = translations[len(input_data):2*len(input_data)]
    translated_outputs = translations[2*len(input_data):]

    # Store the translations in a DataFrame
    translations_df = pd.DataFrame({
        "instruction": translated_instructions,
        "input": translated_inputs,
        "output": translated_outputs
    })

    # Save the translations to a JSON file
    save_translated_subset_to_json(translations_df, file_path="./data/output/BLOOMZ_translations_Vietnamese.json")


# Run the asyncio event loop
start = 40000
end = 51760
subset = True
start_time = time.time()
loop = asyncio.get_event_loop()
## BLOOMZ Model
loop.run_until_complete(main_bloomz(start, end, subset))
## NLLB Model 
#loop.run_until_complete(main(start, end, subset))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")