import json, sys
from PIL import Image
import pytesseract
from openai import OpenAI
import joblib
import pandas as pd
from fuzzywuzzy import fuzz


myconfig = r"--psm 6 --oem 3"
api_key = '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'
client = OpenAI(api_key=api_key)

pytesseract.pytesseract.tesseract_cmd = r'D:\\Tech_Upgrade_Project\\AI\\OCR\\tesseract.exe'

# loading all globals

# existing_manufacturer = pd.read_csv('Manufacturer.csv')
# existing_model = pd.read_csv('models.csv')

# OE_X = joblib.load('OE_X')
LE_Asset = joblib.load('LE_ASSET')
LE_ModelNbr = joblib.load('LE_MODEL')
LE_Manufact = joblib.load('LE_MANUFACT')
rf_classifier = joblib.load('rf_final5')

existing_manufacturer = LE_Manufact.inverse_transform(range(len(LE_Manufact.classes_)))
existing_model = LE_ModelNbr.inverse_transform(range(len(LE_ModelNbr.classes_)))



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
def find_most_similar_value(variable, original_values):
    if not original_values.any():
        return 'UNKNOWN'

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
        return 'UNKNOWN'

    # Get the most similar value from the original values
    most_similar_value = original_values[max_similarity_index]

    return most_similar_value


# ---> getting gpt generated text and processing it to find assetType



def predicted_AssetType(Manufact, model):
    
    if Manufact is None:
        Manufact = 'UNKNOWN'

    if model is None:
        model = 'UNKNOWN'
    

    
    # Create a DataFrame
    X = pd.DataFrame({
        'Manufacturer': [Manufact],
        'ModelNbr': [model]
    })

    special_values = [1234, 'UNKOWN', 'Unknown', 'NA', 'NULL', 'UNKNOWN', 'TBA', 'N/A', 'NOT VISIBLE', '123TEST', 'UNABLE TO LOCATE', 'NO ID', 'NO ACCESS', 'UNKOWN', 'NaN', 'na', 'AS PER PICS','nan','None']

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    X_copy = X.copy()

    # Replace values using .loc to avoid SettingWithCopyWarning
    X_copy.loc[:, 'Manufacturer'] = X_copy['Manufacturer'].replace(special_values, pd.NA)
    X_copy.loc[:, 'ModelNbr'] = X_copy['ModelNbr'].replace(special_values, pd.NA)

    X = X_copy

    # OE_X = joblib.load('OE_X')
    # LE_Asset = joblib.load('LE_Y')

    # X = OE_X.transform(X)

    
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

    model_nbr_matched = 'UNKNOWN'
    manufacturer_matched = 'UNKNOWN'

    for variation in model_nbr_variations:
        if variation in json_data:
            model_nbr = json_data[variation]
            if model_nbr is not None:
                model_nbr = model_nbr.upper()
            model_nbr_matched= find_most_similar_value(model_nbr,existing_model)
            json_data["Model"]=model_nbr_matched
            break

    for variation in manufacturer_variations:
        if variation in json_data:
            manufacturer = json_data[variation]
            if manufacturer is not None:
                manufacturer = manufacturer.upper()
                
            #now match the manufacturer to a valid manufacturer
            manufacturer_matched= find_most_similar_value(manufacturer,existing_manufacturer)
            json_data["Manufacturer"] = manufacturer_matched
            break

    return json.dumps(json_data)       



def process_image(image_path):
    
    # Open the image using Pillow

    # Now you can use the 'pil_image' in your function
    lines, confs, extracted_text = extract_text_from_image(image_path)

    extracted_json = gptText(extracted_text)

    return extracted_json

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
 
    image_path = sys.argv[1] 

    # this function takes stream , converting it to image and then extracting text 
    result = process_image(image_path)

    # this function takes extracted json formatted text and checks for variation of attributes model, manufacturer
    result2 = textProcess(result)

    parsed_data=json.loads(result2)

    #finally assetType is predicted
    parsed_data['AssetType'] = predicted_AssetType(parsed_data["Manufacturer"],parsed_data["Model"])
    print(parsed_data)