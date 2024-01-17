import os
from PIL import Image
import pytesseract
from openai import OpenAI


myconfig = r"--psm 6 --oem 3"
client = OpenAI()

# Set the path to the folder containing images
image_folder_path = "ImageProcessing/check"

# Function to extract text from an image using pytesseract
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config=myconfig)
    return text



def gpt(ocrTXT):
    user_message = {
        "role": "user",
        "content": "from the text find attributes and values" + "\"" + ocrTXT + " \"",
    }
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= [user_message],
            max_tokens=1000
        )

    text = response.choices[0].message.content
    return text


# Loop through all images in the folder
for filename in os.listdir(image_folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg",".JPEG")):
        image_path = os.path.join(image_folder_path, filename)

        # Call the function to extract text from the image
        extracted_text = extract_text_from_image(image_path)

        gptTXT = gpt(extracted_text)

        # Save the text to a text file with the same name as the image
        text_file_path = os.path.splitext(image_path)[0] + ".txt"
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(gptTXT)

        print(f"Text extracted from {filename} and saved to {text_file_path}")
