import json
file_name="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Instruction_finetune_dataset/Belle_open_source_0.5M.json"


# Replace 'your_file.json' with the path to your JSON file
with open(file_name, 'r') as json_file:
    content = json_file.read()

# Split the content by line and remove empty lines
json_objects = [line for line in content.splitlines() if line.strip()]

# Iterate through the JSON objects, load and print them
for index, json_object in enumerate(json_objects):
    try:
        data = json.loads(json_object)
        pretty_data = json.dumps(data, indent=4, sort_keys=True)
        print(f"JSON Object {index + 1}:\n{pretty_data}\n")
    except json.JSONDecodeError as err:
        print(f"Error parsing JSON Object {index + 1}: {err}")
