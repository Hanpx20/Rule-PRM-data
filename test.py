import json

data = json.load(open("data/synthesized_data.json", "r"))
cnt = {}
for scenario, labels in data:
    for key in labels[0].keys():
        if key in cnt:
            cnt[key] += 1
        else:
            cnt[key] = 1

print(cnt)