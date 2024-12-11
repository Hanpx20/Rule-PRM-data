import json
import random


X = json.load(open("data/augmented_data.json", "r"))
Y = []
for idx, content in enumerate(X):
    if idx % 10 == 1:
        Y.append(content)

json.dump(Y, open("data/sample_data.json", "w"), indent=4)