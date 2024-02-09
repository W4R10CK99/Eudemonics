import os, json, csv
from PIL import Image
import pytesseract
from openai import OpenAI
import joblib
import pandas as pd
from fuzzywuzzy import fuzz

myconfig = r"--psm 6 --oem 3"
client = OpenAI()

# loading all globals

existing_manufacturer = pd.read_csv('Manufacturer.csv')
existing_model = pd.read_csv('models.csv')
existing_serial = pd.read_csv('SerialNo.csv')

# OE_X = joblib.load('OE_X')
LE_Asset = joblib.load('LE_ASSET')
LE_Serial = joblib.load('LE_SERIAL')
LE_ModelNbr = joblib.load('LE_MODEL')
LE_Manufact = joblib.load('LE_MANUFACT')
rf_classifier = joblib.load('model/rf_final3')


#---------------------------------------------------------------------------------------------------
# function to calculate confidence score of a word

def calculate_word_confidence(word, lines, confs):
    if not word:
        return 0
    
    word = str(word)
    word_confidence = 0  # Initialize confidence score for the word

    # Iterate over lines to find the exact word and calculate confidence
    for line, conf in zip(lines, confs):
        if line['text'] is not None and isinstance(line['text'], str) and word.lower() in line['text'].lower():
            word_confidence = conf
            break

    return word_confidence

#----------------------------------------------------------------------------------------------------
# function taking a json and returning another json with key as json's data and value confidence score

def process_json_text(json_text, lines, confs):
    # Load JSON text into a Python dictionary
    data = json.loads(json_text)

    # Create a dictionary to store word confidences
    word_confidences = {}

    # Iterate over keys and values in the JSON data
    for key, value in data.items():
        # Calculate confidence for the key and value using the provided lines and confidences
        key_confidence = calculate_word_confidence(key, lines, confs)
        value_confidence = calculate_word_confidence(value, lines, confs)

        # Convert dictionary to string representation
        key_str = json.dumps(key)
        value_str = json.dumps(value)

        # Store confidences in the dictionary
        word_confidences[key_str] = key_confidence
        word_confidences[value_str] = value_confidence

    return word_confidences



def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text_data= pytesseract.image_to_data(img,output_type='data.frame', config=myconfig)
    text_data = text_data[text_data.conf != -1]

    # Convert the 'text' column to strings
    text_data['text'] = text_data['text'].astype(str)

    # Group by relevant columns (e.g., block_num, par_num, line_num)
    lines = text_data.groupby(['block_num', 'par_num', 'line_num'])['text'].apply(lambda x: ' '.join(list(x))).tolist()
    confs = text_data.groupby(['block_num', 'par_num', 'line_num'])['conf'].mean().tolist()

    #now get actual extracted text in original sequence as per image
    lines = []

    # Iterate through the rows of the DataFrame
    for index, row in text_data.iterrows():
        line_num = row['line_num']
        word = row['text']

        # If the lines list is empty or the current line_num is different from the previous one, start a new line
        if not lines or lines[-1]['line_num'] != line_num:
            lines.append({'line_num': line_num, 'text': word})
        else:
            # Append the word to the existing line
            lines[-1]['text'] += ' ' + word

    # Combine the lines into a single variable while preserving the original sequence
    result_text = '\n'.join(line['text'] for line in sorted(lines, key=lambda x: x['line_num'])) 
    return lines, confs, result_text




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
        return 'UNKNOWN'

    if variable and len(variable.split()) > 0:
        variable = variable.split()[0]
    else:
        # Handle the case where variable is empty or has no elements
        variable = 'UNKNOWN'
    
    if variable.lower() == 'johnson':
        variable = 'HITACHI'

    # Calculate similarity using fuzzywuzzy's fuzz.ratio
    similarities = [fuzz.ratio(variable, csv_value) for csv_value in csv_values]

    # Find the index of the maximum similarity
    max_similarity_index = similarities.index(max(similarities))

    if similarities[max_similarity_index] < 70:  # Adjust the threshold as needed
        # If not desirable, return the original value
        
        return 'UNKNOWN'

    
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

    model_nbr_matched = 'UNKNOWN'
    manufacturer_macthed = 'UNKNOWN'
    serial_no_matched = 'UNKNOWN'

    for variation in model_nbr_variations:
        if variation in json_data:
            model_nbr = json_data[variation]
            model_nbr = model_nbr.upper()
            model_nbr_matched= find_most_similar_value(model_nbr,existing_model)
            json_data[variation]=model_nbr_matched
            break

    for variation in manufacturer_variations:
        if variation in json_data:
            manufacturer = json_data[variation]
            manufacturer = manufacturer.upper()
            #now match the manufacturer to a valid manufacturer
            manufacturer_macthed= find_most_similar_value(manufacturer,existing_manufacturer)
            json_data[variation] = manufacturer_macthed
            break

    for variation in serial_no_variations:
        if variation in json_data:
            serial_no = json_data[variation]
            serial_no = serial_no.upper()
            serial_no_matched= find_most_similar_value(serial_no,existing_serial)
            json_data[variation] = serial_no_matched
            break
    
    if model_nbr_matched is None:
        model_nbr_matched = 'UNKNOWN'

    if manufacturer_macthed is None:
        manufacturer_macthed = 'UNKNOWN'
    
    if serial_no_matched is None:
        serial_no_matched = 'UNKNOWN'

    
    # Create a DataFrame
    X = pd.DataFrame({
        'SerialNo': [serial_no_matched],
        'Manufacturer': [manufacturer_macthed],
        'ModelNbr': [model_nbr_matched]
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

    # X = OE_X.transform(X)

    try:
        X['SerialNo'] = LE_Serial.transform(X['SerialNo'])
    except ValueError:
        X['SerialNo']=83686
    
    try:
        X['Manufacturer'] = LE_Manufact.transform(X['Manufacturer'])
    except ValueError:
        X['Manufacturer'] = 335
    
    try:
        X['ModelNbr'] = LE_ModelNbr.transform(X['ModelNbr'])
    except ValueError:
        X['ModelNbr'] = 7364
    
    


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

            # Call the function to extract lines, confs, text from the image
            lines, confs, extracted_text = extract_text_from_image(image_path)

            gptTXT = gptText(extracted_text)
            # print(gptTXT)

            confidence_score = process_json_text(gptTXT, lines, confs)
            finalTXT = predicted_AssetType(gptTXT)
            
            # Save the text to a text file with the same name as the image
            text_file_path = os.path.splitext(image_path)[0] + ".json"
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(finalTXT)

            conf_file_path = os.path.splitext(image_path)[0] + "_confidence.json"
            with open(conf_file_path, "w", encoding="utf-8") as json_file:
                json.dump(confidence_score, json_file, ensure_ascii=False, indent=2)

            print(f"Text extracted from {filename} and saved to {text_file_path}")





def process_directory(base_directory):
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)

        if os.path.isdir(folder_path):
            # If the item in the directory is a folder, process it
            process_folder(folder_path)




# Replace '/path/to/your/directory' with the actual path to your directory
directory_path = 'model/testFinal2'
process_directory(directory_path)