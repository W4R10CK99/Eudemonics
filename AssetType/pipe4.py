import os, json, csv, difflib
from PIL import Image
import pytesseract
from openai import OpenAI
import joblib
import pandas as pd

myconfig = r"--psm 6 --oem 3"
client = OpenAI()

# loading all globals

registered_manufacturer = pd.read_csv('Manufacturer.csv')
OE_X = joblib.load('OE_X')
LE_Asset = joblib.load('LE_Y')
rf_classifier = joblib.load('model/rf_final2_ordinal')


def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config=myconfig)
    return text




def gptText(ocrTXT):
    user_message = {
    "role": "user",
    "content": "from the text find meaningful attributes(Barcode,Model ,Serial,Manufacturer,Date of installation/Manufacturer,Unit of mesurement,Capacity,Eletrical Input KW,Warrenty End Date,Refrigerant,Refrigerant Qty) and their values in json format " + "\"" + ocrTXT + " \"",
    }

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[user_message],
        max_tokens=1000
    )

    text = response.choices[0].message.content
    return text


# Read CSV file and find the most similar value
def find_most_similar_value(variable, csvfile):
    
    reader = csv.reader(csvfile)
    csv_values = [row[0] for row in reader]

    if not csv_values:
        return None
    

    variable = variable.split()[0]
    if(variable=='JOHNSON' or variable=='Johnson'):
        variable='HITACHI'
    # Calculate similarity using difflib's SequenceMatcher
    similarities = [difflib.SequenceMatcher(None, variable, csv_value).ratio() for csv_value in csv_values]

    # Find the index of the maximum similarity
    max_similarity_index = similarities.index(max(similarities))
    if similarities[max_similarity_index] < 0.8:
        # If not desirable, return the original value
        return variable

    # Get the most similar value from the CSV file
    most_similar_value = csv_values[max_similarity_index]

    return most_similar_value


# ---> getting gpt generated text and processing it to find assetType



def predicted_AssetType(txt):
    model_nbr_variations = ['ModelNbr', 'model', 'modelno', 'model no', 'MODEL', 'MODEL NO','Model_No','Model']
    manufacturer_variations = ['Manufacturer', 'manufacturer', 'make','brand','Brand']
    serial_no_variations = ['SerialNo', 'serial', 'serialnumber', 'serialno','Serial_No','Serial Number']

    # Convert the text to a JSON object
    json_data = {}

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
            model_nbr = model_nbr.upper()
            json_data[variation]=model_nbr
            break

    for variation in manufacturer_variations:
        if variation in json_data:
            manufacturer = json_data[variation]
            manufacturer = manufacturer.upper()
            #now match the manufacturer to a valid manufacturer
            manufacturer = find_most_similar_value(manufacturer,registered_manufacturer)
            json_data[variation] = manufacturer
            break

    for variation in serial_no_variations:
        if variation in json_data:
            serial_no = json_data[variation]
            serial_no = serial_no.upper()
            json_data[variation] = serial_no
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

    # OE_X = joblib.load('OE_X')
    # LE_Asset = joblib.load('LE_Y')

    X = OE_X.transform(X)

    # rf_classifier = joblib.load('model/rf_final2_ordinal')
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
            # print(gptTXT)
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
directory_path = 'model/testTemp2'
process_directory(directory_path)