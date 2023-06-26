import json
import string 
import re
from bs4 import BeautifulSoup


# Read the JSON file
with open('suc-khoe.json', 'r') as file:
    data = json.load(file)

with open('suc-khoe_format.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

