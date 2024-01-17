import pytesseract
import PIL.Image
from openai import OpenAI
import json

# Replace 'your_api_key_here' with your actual OpenAI API key
api_key = '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'
myconfig = r"--psm 6 --oem 3"

text = pytesseract.image_to_string(PIL.Image.open('perspective.jpeg'), config=myconfig)

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

user_message = {
    "role": "user",
    "content": "from the text find attributes and their values in format attribute : value " + "\"" + text + " \"",
}

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[user_message],
    max_tokens=1000
)

text = response.choices[0].message.content
print(text)
