import pytesseract as tess
from PIL import Image
from transformers import pipeline

myconfig = r"--psm 6 --oem 3"

text = tess.image_to_string(Image.open('img1.png'),config=myconfig)
print(text)

ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")



results = ner_pipeline(text)

key_value_pairs = {}
for result in results:
    if result['word'].lower() in ['model', 'make', 'type']: 
        key_value_pairs[result['word']] = result['score']

print(key_value_pairs)