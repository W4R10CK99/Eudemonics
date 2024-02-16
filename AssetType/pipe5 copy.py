import json, sys
from PIL import Image
import pytesseract
from openai import OpenAI
import joblib
import pandas as pd
from fuzzywuzzy import fuzz


myconfig = r"--psm 6 --oem 3"
client = OpenAI()

# loading all globals

LE_Serial = joblib.load('encoders/LE_SERIAL')
LE_Asset = joblib.load('encoders/LE_ASSET')
LE_ModelNbr = joblib.load('encoders/LE_MODEL')
LE_Manufact = joblib.load('encoders/LE_MANUFACT')
rf_classifier = joblib.load('model/rf_final8')

existing_manufacturer = LE_Manufact.inverse_transform(range(len(LE_Manufact.classes_)))
existing_model = LE_ModelNbr.inverse_transform(range(len(LE_ModelNbr.classes_)))
existing_serial = LE_Serial.inverse_transform(range(len(LE_Serial.classes_)))


def confidence(json_text, lines, confs):
    # Load JSON text into a Python dictionary
    # Convert the text to a JSON object
    data = {}

    # Assuming txt is some variable containing the JSON string
    try:
        data = json.loads(json_text)
        # Continue with processing json_data
    except json.decoder.JSONDecodeError:
        pass 

    # Create an array to store confidence scores
    word_confidences = {}

    # Iterate over values in the JSON data (ignoring keys)
    for value in data.values():
        # Skip the value if it is not a string
        if not isinstance(value, str):
            continue

        # Initialize confidence score for the value
        value_confidence = 0

        # Iterate over lines to find the exact value and calculate confidence
        for line, conf in zip(lines, confs):
            if line['text'] is not None and isinstance(line['text'], str) and value.lower() in line['text'].lower():
                value_confidence = conf
                break

        # Append the confidence score to the array
        if(value_confidence>0):
            value_str = json.dumps(value)
            word_confidences[value_str] = value_confidence


    return word_confidences

#----------------------------------------------------------------------------------------------------

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
    "content": "from the text find meaningful attributes(Model ,Serial,Manufacturer,Date of installation/Manufacturer,Unit of measurement,Capacity,Eletrical Input KW,Warranty End Date,Refrigerant,Refrigerant Qty) and their values in json format " + "\"" + ocrTXT + " \"",
    }

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[user_message],
        max_tokens=1000
    )

    text = response.choices[0].message.content
    return text

def find_most_similar_serial(variable, original_values):
    if not original_values.any():
        return 'UNKNOWN'

    # variable = variable.replace(' ','')
    if variable and len(variable.split()) > 0:
        variable = variable
    else:
        # Handle the case where variable is empty or has no elements
        variable = 'UNKNOWN'

    # Calculate similarity using fuzzywuzzy's fuzz.ratio
    similarities = [fuzz.ratio(variable.upper(), value.upper()) for value in original_values]

    # Find the index of the maximum similarity
    max_similarity_index = similarities.index(max(similarities))

    if similarities[max_similarity_index] < 0.5:  # Adjust the threshold as needed
        # If not desirable, return the original value
        return 'UNKNOWN'

    # Get the most similar value from the original values
    most_similar_value = original_values[max_similarity_index]

    return most_similar_value



# Read CSV file and find the most similar value
def find_most_similar_value(variable, original_values):
    if not original_values.any():
        return 'UNKNOWN','UNKNOWN'

    if variable and len(variable.split()) > 0:
        variable = variable.split()[0]
    else:
        # Handle the case where variable is empty or has no elements
        variable = 'UNKNOWN'

    if variable.upper() == 'JOHNSON':
        variable = 'HITACHI'

    # Calculate similarity using fuzzywuzzy's fuzz.ratio
    similarities = [fuzz.ratio(variable.upper(), value.upper()) for value in original_values]

    # Find the index of the maximum similarity
    max_similarity_index = similarities.index(max(similarities))

    if similarities[max_similarity_index] < 0.5:  # Adjust the threshold as needed
        # If not desirable, return the original value
        return 'UNKNOWN',variable

    # Get the most similar value from the original values
    most_similar_value = original_values[max_similarity_index]

    return most_similar_value,variable


# ---> getting gpt generated text and processing it to find assetType



def predicted_AssetType(Serial,Manufact, model):
    if Serial is None:
        Serial = 'UNKNOWN'
    
    if Manufact is None:
        Manufact = 'UNKNOWN'

    if model is None:
        model = 'UNKNOWN'
    

    
    # Create a DataFrame
    X = pd.DataFrame({
        'SerialNo': [Serial],
        'Manufacturer': [Manufact],
        'ModelNbr': [model]
    })

    special_values = [1234,'NOT SHOWN','SERIAL','NO SERIAL','SERIAL FADED', 'UNKOWN','REFER TO PIC','FADED LABEL','N.A.', 'UNKOWN', 'Unknown', 'NA', 'NULL', 'UNKNOWN', 'TBA', 'N/A', 'NOT VISIBLE', '123TEST', 'UNABLE TO LOCATE', 'NO ID', 'NO ACCESS', 'UNKOWN', 'NaN', 'na', 'AS PER PICS','nan','None']

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

    X['SerialNo'] = X['SerialNo'].str.replace(' ', '')

    try:
        X['SerialNo'] = LE_Serial.transform(X['SerialNo'])
    except ValueError:
        X['SerialNo'] = 27278

    try:
        X['Manufacturer'] = LE_Manufact.transform(X['Manufacturer'])
    except ValueError:
        X['Manufacturer'] = 286
    
    try:
        X['ModelNbr'] = LE_ModelNbr.transform(X['ModelNbr'])
    except ValueError:
        X['ModelNbr'] = 7364
    
    


    y_pred_encoded = rf_classifier.predict(X)
    y_actual = LE_Asset.inverse_transform(y_pred_encoded)
    

    return y_actual[0]


def textProcess(txt):
    model_nbr_variations = ['ModelNbr', 'model', 'modelno', 'model no', 'MODEL', 'MODEL NO','Model_No','Model']
    manufacturer_variations = ['Manufacturer', 'manufacturer', 'make','brand','Brand']
    serial_variations = ['Serial','SERIAL','SERIAL NO','Serial No','Serial NO','Serial Number','Serial No.','SERIAL NO.','SERIAL NUMBER']

    # Convert the text to a JSON object
    json_data = {}

    # Assuming txt is some variable containing the JSON string
    try:
        json_data = json.loads(txt)
        # Continue with processing json_data
    except json.decoder.JSONDecodeError:
        pass 

    # Extract attributes
    serial_nbr = None
    model_nbr = None
    manufacturer = None

    serial_matched = 'UNKNOWN'
    model_nbr_matched = 'UNKNOWN'
    manufacturer_matched = 'UNKNOWN'

    for variation in serial_variations:
        if variation in json_data:
            serial_nbr = json_data[variation]
            if serial_nbr is not None:
                serial_nbr = serial_nbr.upper()
            serial_matched= find_most_similar_serial(serial_nbr,existing_serial)
            del json_data[variation]
            json_data["Serial"]=serial_nbr
            break

    for variation in model_nbr_variations:
        if variation in json_data:
            model_nbr = json_data[variation]
            if model_nbr is not None:
                model_nbr = model_nbr.upper()
            model_nbr_matched,original_model= find_most_similar_value(model_nbr,existing_model)
            del json_data[variation]
            json_data["Model"]=original_model
            break

    for variation in manufacturer_variations:
        if variation in json_data:
            manufacturer = json_data[variation]
            if manufacturer is not None:
                manufacturer = manufacturer.upper()
                
            #now match the manufacturer to a valid manufacturer
            manufacturer_matched,original_manufact= find_most_similar_value(manufacturer,existing_manufacturer)
            del json_data[variation]
            json_data["Manufacturer"] = manufacturer_matched
            break

    return json.dumps(json_data),serial_matched,manufacturer_matched,model_nbr_matched      



def process_image(image_path):
    
    #extract text, confidence score variables
    lines, confs, extracted_text = extract_text_from_image(image_path)

    #meaningful value from extracted text
    extracted_json = gptText(extracted_text)

    return extracted_json, lines, confs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
 
    image_path = sys.argv[1] 

    # this function takes stream , converting it to image and then extracting text 
    result, lines, confs = process_image(image_path)
    word_confidences = confidence(result,lines,confs)
    print(result)
    print(word_confidences)
    # this function takes extracted json formatted text and checks for variation of attributes model, manufacturer
    result2,serial_matched,manufacturer_matched,model_nbr_matched = textProcess(result)

    parsed_data=json.loads(result2)
    result = json.loads(result)
    # Get the confidence scores for "Serial," "Manufacturer," and "Model"
    serial_confidence = word_confidences.get(f'"{result["Serial"]}"', 0)
    manufacturer_confidence = word_confidences.get(f'"{result["Manufacturer"]}"', 0)
    model_confidence = word_confidences.get(f'"{result["Model"]}"', 0)

    min_confidence = min(serial_confidence, manufacturer_confidence, model_confidence)
    if(min_confidence < 20):
        parsed_data['AssetType'] = ""
    
    else:
        #finally assetType is predicted
        parsed_data['AssetType'] = predicted_AssetType(serial_matched,manufacturer_matched,model_nbr_matched)
    
    
    print(parsed_data)