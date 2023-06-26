from datasets import load_dataset
import json
dataset = load_dataset("databricks/databricks-dolly-15k", )
df_train = dataset['train'].to_pandas()

# Save df_test to JSON
df_train.to_json("Dolly_train_data.json", orient="records", lines=True)



## if the category is close Q&A, then the response is the answer "context"

with open('Dolly_train_data.json', 'r') as json_file:

    data = json_file.readlines()

# Extract conversation values and create a new list of dictionaries
new_data = []
for row in data:

    obj = json.loads(row)
    category = obj['category']
    if category == 'closed_qa' and len(obj['context']) > len(obj['response']):
        prompt = obj['instruction']
        response = obj['context']
        new_data.append({"prompt": prompt, "response": response})
    else:
        conversation = obj['instruction']
        #breakpoint()
        prompt = obj["instruction"]
        response = obj["response"] 
        new_data.append({"prompt": prompt, "response": response})

# Save the new list of dictionaries to a JSON file
with open('Dolly_train_data_format.json', 'w') as json_file:
    json.dump(new_data, json_file)


# Load the new JSON file
with open('Dolly_train_data_format.json', 'r') as json_file:
    new_data = json.load(json_file)
    #new_data = json_file.readlines()
# Count the number of total rows
total_rows = len(new_data)

# Print the total number of rows
print("Total rows:", total_rows)