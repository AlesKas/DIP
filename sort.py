import os
import json

with open('output/information_criteria/AR/criterions.json') as json_file:
    data = json.load(json_file)

dir = 'output/information_criteria/AR/new/'

for file in os.listdir(dir):
    full_file_name = dir + file
    with open(full_file_name, 'r') as json_file:
        new_data = json.load(json_file)
    data.append(new_data)

data = sorted(data, key=lambda x : x[0])

with open('crit.json', 'w') as json_file:
    json.dump(data, json_file) 