import os, json
from PIL import Image
import pytesseract
from openai import OpenAI
import joblib
import pandas as pd

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

def predicted_AssetType(txt):
    model_nbr_variations = ['ModelNbr', 'model', 'modelno', 'model no', 'MODEL', 'MODEL NO']
    manufacturer_variations = ['Manufacturer', 'manufacturer', 'make']
    serial_no_variations = ['SerialNo', 'serial', 'serialnumber', 'serialno']

    # Convert the text to a JSON object
    json_data

    # Assuming txt is some variable containing the JSON string
    try:
        json_data = json.loads(txt)
        # Continue with processing json_data
    except json.decoder.JSONDecodeError:
        pass 

    # Extract attributes
    model_nbr = None
    manufacturer = None
    serial_no = None

    for variation in model_nbr_variations:
        if variation in json_data:
            model_nbr = json_data[variation]
            break

    for variation in manufacturer_variations:
        if variation in json_data:
            manufacturer = json_data[variation]
            break

    for variation in serial_no_variations:
        if variation in json_data:
            serial_no = json_data[variation]
            break

    # Create a DataFrame
    X = pd.DataFrame({
        'SerialNo': [serial_no],
        'Manufacturer': [manufacturer],
        'ModelNbr': [model_nbr]
    })

    special_values = [1234, 'UNKOWN', 'Unknown', 'NA', 'NULL', 'UNKNOWN', 'TBA', 'N/A', 'NOT VISIBLE', '123TEST', 'UNABLE TO LOCATE', 'NO ID', 'NO ACCESS', 'UNKOWN', 'NaN', 'na', 'AS PER PICS','nan','None']

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    X_copy = X.copy()

    # Replace values using .loc to avoid SettingWithCopyWarning
    X_copy.loc[:, 'SerialNo'] = X_copy['SerialNo'].replace(special_values, pd.NA)
    X_copy.loc[:, 'Manufacturer'] = X_copy['Manufacturer'].replace(special_values, pd.NA)
    X_copy.loc[:, 'ModelNbr'] = X_copy['ModelNbr'].replace(special_values, pd.NA)

    X = X_copy

    OE_X = joblib.load('OE_X')
    LE_Asset = joblib.load('LE_Y')

    X = OE_X.transform(X.astype(str))

    rf_classifier = joblib.load('rf_final2_ordinal')
    y_pred_encoded = rf_classifier.predict(X)
    y_actual = LE_Asset.inverse_transform(y_pred_encoded)
    
    # Append the predicted AssetType to the original JSON data
    json_data['PredictedAssetType'] = y_actual[0]

    # Convert the updated JSON data back to a string
    updated_txt = json.dumps(json_data)

    return updated_txt


def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg",".JPEG")):
            image_path = os.path.join(folder_path, filename)

            # Call the function to extract text from the image
            extracted_text = extract_text_from_image(image_path)

            gptTXT = gptText(extracted_text)

            finalTXT = predicted_AssetType(gptTXT)
            
            # Save the text to a text file with the same name as the image
            text_file_path = os.path.splitext(image_path)[0] + ".json"
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(finalTXT)

            print(f"Text extracted from {filename} and saved to {text_file_path}")

def process_directory(base_directory):
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)

        if os.path.isdir(folder_path):
            # If the item in the directory is a folder, process it
            process_folder(folder_path)

# Replace '/path/to/your/directory' with the actual path to your directory
directory_path = 'testFinal'
process_directory(directory_path)