from datasets import load_dataset
import json


dataset = load_dataset("GAIR/lima", )
df_train = dataset['train'].to_pandas()
df_test = dataset['test'].to_pandas()
# # Convert the dataset dictionary to JSON format
# json_data = json.dumps(dataset)

# # Save the JSON data to a file
# with open("sample_data.json", "w") as json_file:
#     json_file.write(json_data)

# Save df_train to JSON
df_train.to_json("train_data.json", orient="records", lines=True)

# Save df_test to JSON
df_test.to_json("test_data.json", orient="records", lines=True)


import json

# # Read the JSON file
# with open('/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/test_data.json', 'r') as json_file:
#     #data = json.load(json_file)
#     data = json_file.read()

# # Extract conversation values and create a new list of dictionaries
# # Parse the JSON objects
# rows = data.split(',')
# new_data = []
# for row in data['conversations']:
#     prompt = row[0]
#     response = row[1]
#     new_data.append({"prompt": prompt, "response": response})

# # Print the new list of dictionaries
# # for item in new_data:
# #     print(item)

# # Save the new list of dictionaries to a JSON file
# with open('lima_format_structure.json', 'w') as json_file:
#     json.dump(new_data, json_file)


import json


# Read the JSON file
#with open('your_file.json', 'r') as json_file:
with open('/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/train_data.json', 'r') as json_file:

    data = json_file.readlines()

# Extract conversation values and create a new list of dictionaries
new_data = []
for row in data:
    obj = json.loads(row)
    conversation = obj['conversations']
    #breakpoint()
    prompt = conversation[0]
    response = conversation[1] if len(conversation) > 1 else ""
    new_data.append({"prompt": prompt, "response": response})

# Save the new list of dictionaries to a JSON file
with open('lima_format_structure_train.json', 'w') as json_file:
    json.dump(new_data, json_file)


# Load the new JSON file
with open('lima_format_structure_train.json', 'r') as json_file:
    new_data = json.load(json_file)
    breakpoint()

# Count the number of total rows
total_rows = len(new_data)

# Print the total number of rows
print("Total rows:", total_rows)