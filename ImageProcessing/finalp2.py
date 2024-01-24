import pytesseract
import PIL.Image
from openai import OpenAI
import json

# Replace 'your_api_key_here' with your actual OpenAI API key
myconfig = r"--psm 6 --oem 3"

text = pytesseract.image_to_string(PIL.Image.open('ImageProcessing\check3\Img.JPEG'), config=myconfig)

# Initialize OpenAI client with the API key
client = OpenAI()

user_message = {
    "role": "user",
    "content": "from the text find attributes and their values in json format " + "\"" + text + " \"",
}

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[user_message],
    max_tokens=1000
)

text = response.choices[0].message.content
print(text)
