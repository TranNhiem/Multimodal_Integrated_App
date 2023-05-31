
import os 
# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer


checkpoint = "bigscience/bloomz-3b" #mT0 architecture #"bigscience/mt0-xl"; "bigscience/mt0-xxl"
weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLOOMZ/"
# Create the weight_path if it is not exist 
if not os.path.exists(weight_path): 
    os.makedirs(weight_path)
# Initialize the tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(checkpoint, )#cache_dir=weight_path,
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
inputs = tokenizer.encode("Translate to Chinese: {1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. Exercise regularly to keep your body active and strong. Get enough sleep and maintain a consistent sleep schedule.}", return_tensors="pt",truncation=True, max_length=400).to("cuda")
outputs = model.generate(inputs, max_length=400)
print(tokenizer.decode(outputs[0]))


