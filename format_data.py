import json

with open("VOA/data.jsonl", "w") as f:
    for i in range(29284):
        with open(f"VOA/title/{i}_metadata.txt", "r") as f1, open(f"VOA/txt/{i}.txt", "r") as f2:
            title = f1.read()
            text = f2.read()
            data = {"id": i, "title": title, "text": text}
            f.write(json.dumps(data) + "\n")
