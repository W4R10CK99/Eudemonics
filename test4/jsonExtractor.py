import json

text = """
        Possible attributes and their values extracted from the text:

- Model: RAS-100HNCELW
- Serial No.: ANOOO111
- Year: 2022
- Barcode: AM202381387
- Manufacturer: Johnson Controls-Hitachi Air Conditioning Wuhu Co. Ltd
- Refrigerant: R410A
- Weight: 197 kg
- Power Supply: 380-415V 3N~50Hz
- Capacity (Cooling/Heating): 280/315 kW
- Waterproof Class: IPX4
- Air Volume: 185 m/min
- Power Input (Cooling/Heating): 781/736 W
- Current (Cooling/Heating): 192/42 A
- Rated Power Input (Max): 124 kW
- Rated Current (Max): 21.0 A

Possible JSON format:

{
  "Model": "RAS-100HNCELW",
  "Serial No.": "ANOOO111",
  "Year": 2022,
  "Barcode": "AM202381387",
  "Manufacturer": "Johnson Controls-Hitachi Air Conditioning Wuhu Co. Ltd",
  "Refrigerant": "R410A",
  "Weight": "197 kg",
  "Power Supply": "380-415V 3N~50Hz",
  "Capacity": {
    "Cooling": "280 kW",
    "Heating": "315 kW"
  },
  "Waterproof Class": "IPX4",
  "Air Volume": "185 m/min",
  "Power Input": {
    "Cooling": "781 W",
    "Heating": "736 W"
  },
  "Current": {
    "Cooling": "192 A",
    "Heating": "42 A"
  },
  "Rated Power Input (Max)": "124 kW",
  "Rated Current (Max)": "21.0 A"
}
        """

# Extract JSON part
start_index = text.find("{")
end_index = text.rfind("}") + 1
json_part = text[start_index:end_index]
json_data = json.loads(json_part)

# Print the extracted JSON
text2 = json.dumps(json_data, indent=2)
print(text2)