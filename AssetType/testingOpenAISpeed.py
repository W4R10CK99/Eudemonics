from openai import OpenAI
import numpy as np, pandas as pd

client = OpenAI()

def gptText(ocrTXT):
    user_message = {
    "role": "user",
    "content": "from the text find meaningful attributes(Barcode,Model ,Serial,Manufacturer,Date of installation/Manufacturer,Unit of mesurement,Capacity,Eletrical Input KW,Warrenty End Date,Refrigerant,Refrigerant Qty) and their values in json format" + "\"" + ocrTXT + " \"",
    }

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[user_message],
        max_tokens=1000
    )

    text = response.choices[0].message.content
    return text

text = """. a
: . > c
*~Y ~ ~<
= “~ i Ne * ~~ -
* 4 =
3 ~ i
s é ’ : inl ~
eee — —
o ~~
i ~~ i
eed “a Me!
= i ‘ a ag
* 7 : - i :
“ : ——- — ~
—— Quality Ventilation Products 4% >
——— | . - \ ’
— Australian Made s sie 50 Jae
=_—_— eee ROIg eS Z3y =
FANTECH 9 [mtg code CPECSS4°HP SSieepass Lien ieee
EE ere
[Date 02-Jun-16 | javoice Moz77036._—|Serial 662031 SS ae See Sores ™
Customer HEATCRAFT AUSTRALIA PTY LTD SSeS eset ey LPs Bee
re So Ee eriore ao" Bake,
Project See Seep
- ___ ee BRE re wits
Unit Marking ee ie
ee PETE cog eg Barges
Order No. 950257 — 4c BE! area its
oes Serio cgapag ie are
[kW _0.26(Shaft) | FLA 18 TOM s
: TE cs ees ead Corcoran
Motor ELTA Voltage 240V 1Ph 50Hz See gee
Hajar installation & Maintenance Instrupnons for greasing LL Fe AI are et
Australia: Fantech Pty.Ltd. J) wn TE A | AAeorn es
(03) 9554 7845 ) | WE AIRPOR
- ® New Zealend: Fe itech Ltd. | i a | \\\ \\N\WMY
3 (09) 444 6:66 ) Kl \ | le A\\\
| wwwfaitechiomau 0 000562 rte il APAM013789
% - es Baie S 1S Lael Ereaa ee eae -
a eee"""

result = gptText(text)

print(result)