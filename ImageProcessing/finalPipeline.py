"""
Page segmentation modes: 
O Orientation and script detection (OSD) only
1 Automatic page segmentation with OSD. 
2 Automatic page segmentation, but no OSD, or OCR.
3 Fully automatic page segmentation, but no OSD. (Default)
4 Assume a single column of text of variable sizes.
5 Assume a single uniform block of vertically aligned text.
6 Assume a single uniform block of text
7 Treat the image as a single text line.
8 Treat the image as a single word.
9 Treat the image as a single word in a circle.
10 Treat the image as a single character.
11 Sparse text. Find as much text as possible in no particular order.
12 Sparse text with OSD.
13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract—specific.
"""

import pytesseract
import PIL.Image
from openai import OpenAI
import json

myconfig = r"--psm 6 --oem 3"

text = pytesseract.image_to_string(PIL.Image.open('perspective.jpeg'),config=myconfig)

client = OpenAI()
user_message = {
    "role": "user",
    "content": "from the text find attributes and values" + "\"" + text + " \"",
}
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages= [user_message],
        max_tokens=1000
    )

text = response.choices[0].message.content
print(text)

# interest_keys = [
#     "Barcode",
#     "Model",
#     "Serial No.",
#     "Manufacturer",
#     "Date of installation/Manufacturer",
#     "Unit of measurement",
#     "Capacity",
#     "Electrical Input KW",
#     "Warranty End Date",
#     "Refrigerant",
#     "Refrigerant Qty",
# ]

# # Split the text into lines and filter specific key-value pairs
# key_value_pairs = [line.split(':') for line in text.split('\n') if ':' in line]
# filtered_data_dict = {
#     key.strip().lower(): value.strip() for key, value in key_value_pairs if key.strip().lower() in map(str.lower, interest_keys)
# }

# # Convert the filtered dictionary to JSON format
# json_data = json.dumps(filtered_data_dict, indent=2)

# # Save the JSON data to a file
# with open('output.json', 'w') as json_file:
#     json_file.write(json_data)