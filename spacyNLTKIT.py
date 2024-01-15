import spacy

def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    entities = {}
    current_entity = None

    for token in doc:
        if token.ent_type_:
            current_entity = token.ent_type_
            entities[current_entity] = [token.text]
        elif current_entity:
            entities[current_entity].append(token.text)

    return {key: ' '.join(value) for key, value in entities.items()}

# Example usage
input_paragraph = """MODEL _GPC1460H41EC SERIAL NO. 1609315090
AC VOLTS 206/230 PH 1 HERTZ 60 _
USE COPPER CONDUCTORS ONLY
VOLTAGE RANGE MAX 263 MIN 1497 MIN CIRCUIT AMP8 49
COMPRES8OR LA 26.4 LAA 134 FUSE MAX AMPS 60
COND MOTOR FLA 1.4 HP 1/4 CHARGE 103 07. RA10A
BLOWER MOTOR FLA 6.4 HP 3/4
FACTORY TEST PRESSURE PSIG LOW SIDE 236 _ HIGH SIDE _450
MAXIMUM HACR AMPS 60 = PART NO. _ 9140004316
CIRCUIT BREAKER
THI8 EQUIPMENT SUITABLE.FOR OUTDOOR USE ONLY
WHEN INSTALLING ONE OF THE ACCESSORY HEAT KITS. =
— PERMANTELY IDENTIFY THE MODEL ON THI8 PLATE
AND MARK THE SINGLE POINT WIRING KIT IF INSTALLED.

— 0” CLEARANCE TO COMBUSTIBLE ane ~

= M OUTLET AIR TEMPERA

“ rian HEAT KITS ARE RATED AT 240 VAC/1 PH/ 60 _ HERTZ"""
result = extract_entities(input_paragraph)
print(result)