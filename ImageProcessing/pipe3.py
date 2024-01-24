import os, json
from PIL import Image
import pytesseract
from openai import OpenAI

myconfig = r"--psm 6 --oem 3"
client = OpenAI()


def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config=myconfig)
    return text

def gptText(ocrTXT):
    user_message = {
    "role": "user",
    "content": "from the text find meaningful attributes and their values in json format " + "\"" + ocrTXT + " \"",
    }

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[user_message],
        max_tokens=1000
    )

    text = response.choices[0].message.content
    return text


def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg",".JPEG")):
            image_path = os.path.join(folder_path, filename)

            # Call the function to extract text from the image
            extracted_text = extract_text_from_image(image_path)

            gptTXT = gptText(extracted_text)

            # Save the text to a text file with the same name as the image
            text_file_path = os.path.splitext(image_path)[0] + ".json"
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(gptTXT)

            print(f"Text extracted from {filename} and saved to {text_file_path}")

def process_directory(base_directory):
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)

        if os.path.isdir(folder_path):
            # If the item in the directory is a folder, process it
            process_folder(folder_path)

# Replace '/path/to/your/directory' with the actual path to your directory
directory_path = 'test4'
process_directory(directory_path)