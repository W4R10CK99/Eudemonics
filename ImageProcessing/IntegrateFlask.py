from flask import Flask, request, jsonify, send_file
import pytesseract
from PIL import Image
from openai import OpenAI
import os
import json

app = Flask(__name__)

myconfig = r"--psm 6 --oem 3"
client = OpenAI()

def perform_ocr(image_file):
    image = Image.open(image_file)
    extracted_text = pytesseract.image_to_string(image, config=myconfig)
    return extracted_text

def send_to_openai(text):
    user_message = {
        "role": "user",
        "content": "from the text find attributes and values" + "\"" + text + " \"",
    }
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[user_message],
        max_tokens=1000
    )
    return response.choices[0].message.content

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Perform OCR using Tesseract on the uploaded image
        extracted_text = perform_ocr(file)

        # Send extracted text to OpenAI API for further processing
        openai_response = send_to_openai(extracted_text)

        # Save the result as a JSON file
        result_filename = 'result.json'
        result_path = os.path.join(app.root_path, result_filename)
        with open(result_path, 'w') as result_file:
            json.dump({'result': openai_response}, result_file)

        # Provide a link to download the JSON file
        download_link = f'/download/{result_filename}'
        return jsonify({'download_link': download_link}), 200

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    # Provide the option to download the JSON file
    return send_file(filename, as_attachment=True)


@app.route('/')
def welcome():
    return 'Welcome to the OCR and OpenAI API integration!'

if __name__ == '__main__':
    app.run(debug=True)

