import json 
import os 
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def merge_json_files():
    """
    Merge multiple JSON files into a single JSON file.
    Checking rows with 'instruction' and 'output' keys are exist in the JSON file.
    """
    merged_data = []

    # Find JSON files with names matching the pattern
    pattern = r'.*checkpoint_(\d+)\.json'
    json_directory = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/Dolly/"

    file_list = os.listdir(json_directory)
    filtered_files = sorted([file for file in file_list if re.match(pattern, file)], key=lambda x: int(re.match(pattern, x).group(1)))

    for file_name in filtered_files:
        file_path = os.path.join(json_directory, file_name)
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                for item in data:
                    if "prompt" in item and "response" in item:
                        instruction = item.get("prompt", "")
                        output = item.get("response", "")

                        if isinstance(instruction, list):
                            instruction = instruction[0]

                     
                        if isinstance(output, list):
                            output = output[0]

                        new_entry = {
                            'prompt': instruction,
                            'response': output
                        }

                        # Check if the new entry already exists in merged_data
                        if new_entry not in merged_data:
                            merged_data.append(new_entry)

            except json.JSONDecodeError as e:
                print(f"Error parsing file: {file_path}. Error message: {str(e)}")

    # Write merged data to a new file
    output_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/Dolly/Dolly_translate_GPT_4_all.json"
    with open(output_file_path, 'w', encoding='utf-8' ) as file:
        json.dump(merged_data, file,  ensure_ascii=False)

    print(f"Merged data has been written to {output_file_path}")

# Call the function to merge the JSON files
# merge_json_files()
#file_path= "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/Dolly/Dolly_translate_GPT_4_all.json"
file_path="/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_gpt4all_merged_data_.json"
with open(file_path, 'r') as file:
    data = json.load(file)
    print(len(list(data)))
miss_translated = []
good_translated = []
for item in data:
    #print(item)
    prompt= item.get("prompt", "")
    response= item.get("response", "")
   
    # Skip if prompt or response is empty or not a string
    if not isinstance(prompt, str) or not isinstance(response, str) or prompt.strip() == "" or response.strip() == "":
        continue
    try: 
        language_prompt = detect(prompt)
        language_response = detect(response)
        if language_prompt != 'en' and language_response != 'en':
            print("prompt: ", prompt)
            print("response: ", response)
            print("language_prompt: ", language_prompt)
            print("language_response: ", language_response)
            print("===========================================================")
            good_translated.append(item)
        else: # save to new file
            miss_translated.append(item)
    
    except LangDetectException as e:
        print("Error detecting language:", str(e))
        continue

 # Write merged data to a new file
#output_file_path = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/Dolly/Dolly_miss_translate.json"
output_file_path="/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/Alpaca_GPT4all_miss_translate.json"
# # Write miss_translated data to a new file
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(miss_translated, file, ensure_ascii=False)
#output_file_path_ = "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/Translate_modules/Dolly/Dolly_goodtranslate.json"
output_file_path_="/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/Alpaca_GPT4all_good_translate.json"
# Write miss_translated data to a new file
with open(output_file_path_, 'w', encoding='utf-8') as file:
    json.dump(good_translated, file, ensure_ascii=False)

print("Miss-translated items saved to", output_file_path)
