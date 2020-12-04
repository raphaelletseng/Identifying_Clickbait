import json

texts = {}
titles = {}
labels = {}
allScores = []

with open('instances.jsonl') as file:
    for line in file.readlines():
        d = json.loads(line)
        texts[d['id']] = " ".join([p for p in d['targetParagraphs']])
        titles[d['id']] = d['targetTitle']

with open('truth.jsonl') as file:
    for line in file.readlines():
        d = json.loads(line)
        labels[d['id']] = d['truthMean']
        for score in d['truthJudgments']:
            allScores.append(score)

print(len(labels), len(texts))

