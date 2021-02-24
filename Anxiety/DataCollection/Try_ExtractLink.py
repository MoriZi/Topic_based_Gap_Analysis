import json

links=[]
# Opening JSON file
f = open('/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/JSON/ranxiety11.json',)

# returns JSON object as a dictionary
data = json.load(f)

# Iterating through the json
for i in data['data']:
    #print(i['full_link'])
    links.append(i['full_link'])

# Closing file
f.close()

print(links)
