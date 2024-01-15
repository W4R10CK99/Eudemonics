from openai import OpenAI
import os

client = OpenAI()
user_message = {
    "role": "user",
    "content": """from the text find attributes and values \"MODEL _GPC1460H41EC SERIAL NO. 1609315090
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

        = M OUTLET AIR TEMPERA\"""",
}
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages= [user_message],
        max_tokens=1000
    )

print(response['choices'][0]['message']['content'])