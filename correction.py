import json 

with open('patient_29_labels.json', 'r') as file:
    data = json.load(file)


for d in data['labels']:
    d['start'] = d['start'] * 250 / 128
    d['end'] = d['end'] * 250 / 128

json_str = json.dumps(data, indent=1)
with open("test.json", "w") as f:
    f.write(json_str)