import csv
import json
from underthesea import ner


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert CSV file to JSON format and save it to a file.

    Args:
        csv_file_path (str): Path to the CSV file.
        json_file_path (str): Path to the output JSON file.
    """
    # Open the CSV file for reading
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        # Read the CSV data
        csv_data = csv.DictReader(csv_file)

        # Convert CSV data to a list of dictionaries
        json_data = list(csv_data)

        # Write the JSON data to a file
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False)

# Specify the path of the CSV file
csv_file_path = './output.csv'

# Specify the path of the output JSON file
json_file_path = './vihealth_QA_output.json'

# Convert CSV to JSON
csv_to_json(csv_file_path, json_file_path)


# Read the JSON file
with open('vihealth_QA_output.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Remove unnecessary strings and format the data
format_data = []
for item in json_data:
    prompt = item.get("Question", "")
    response = item.get("Answer", "")

    if not isinstance(prompt, str) or not isinstance(response, str) or prompt.strip() == "" or response.strip() == "":
        continue

    # Strings to remove from the prompt and response
    question_removed_string = [
        "Chào bác sĩ ạ!", "Chào bác sĩ.", "bác sĩ cho hỏi,", "Chào bác sĩ ạ.", "Chào BS,", "Chào BS ạ,", "Xin chào bác sĩ ạ.",
        "Thưa bác sĩ!", "Thưa bác sĩ,", "Xin chào bác sĩ!", "Kính chào bác sĩ!", "Xin chào bác sĩ,", "Bác sĩ ơi,",
        "Xin chào bác sĩ ạ!", "Chào bác sĩ,", "Chào bác sĩ!", "Chào Bác sĩ."
    ]

    answer_removed_string = [
        "bác sĩ xin được giải đáp như sau:", "Bác sĩ,", "bác sĩ xin giải đáp như sau:", "Bác sĩ xin giải đáp như sau:"
    ]

    # Remove unnecessary strings from the prompt
    for string in question_removed_string:
        prompt = prompt.replace(string, "")

    # Remove unnecessary strings from the response
    for string in answer_removed_string:
        response = response.replace(string, "")

    def remove_names(text):
        """
        Remove named entities (person and organization names) from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with named entities removed.
        """
        entities = ner(text)
        for entity in entities:
            if entity[3] in ["PER", "ORG"]:
                text = text.replace(entity[0], "")
        return text

    # Remove named entities from the prompt and response
    prompt = remove_names(prompt)
    response = remove_names(response)

    structure_data = {
        "prompt": prompt,
        "response": response
    }
    format_data.append(structure_data)

# Save the formatted data to a JSON file
with open("vihealth_QA_format.json", 'w', encoding='utf-8') as file:
    json.dump(format_data, file, ensure_ascii=False)
