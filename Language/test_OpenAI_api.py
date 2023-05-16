import os
import openai
openai.organization = "org-PVVobcsgsTm9RT8Ez5DubzbX"
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Model.list())


prompt = "Translate this into Vietnamese: \n\ninstruction: Give three tips for staying healthy.,input: "",output: 1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.\n\n1"
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0,
  max_tokens=1000,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
translated_text = response.choices[0].text.strip()
print(translated_text.split('\n')[-1].strip())