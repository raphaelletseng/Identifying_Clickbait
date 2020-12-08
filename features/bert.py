from keybert import KeyBERT
import json
import scipy
import nltk
from sentence_transformers import SentenceTransformer
#Requires
#pip install -U sentence-transformers
#pip install keybert

nltk.download('stopwords')
vectorizer = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
model = KeyBERT('distilbert-base-nli-mean-tokens')

def porterStem(paragraph):
    stem = nltk.stem.porter.PorterStemmer().stem
    data = []
    for word in paragraph.split():
        if(word in nltk.corpus.stopwords.words('english')):
            continue
        data.append(stem(word))
        
    return " ".join(data)

texts = []
titles = []
labels = []
dictLabels = {}

with open('../data/truth.jsonl') as file:
    for line in file.readlines():
        d = json.loads(line)
        dictLabels[d['id']] = d['truthMean']
index = 0            
with open('../data/instances.jsonl') as file:
    for line in file.readlines():
        d = json.loads(line)
        texts.append("\n ".join([porterStem(p) for p in d['targetParagraphs']]))
        titles.append(d['targetTitle'])
        labels.append(dictLabels[d['id']])
        index+=1
        if(index==10):
            break

sim = scipy.spatial.distance.cosine
n = 2
k = len(titles[n].split())

print(titles[n], "\nClickbait average: ", labels[n])

keyWords = " ".join(model.extract_keywords(texts[n],top_n=k))
u = vectorizer.encode(titles[n])
v = vectorizer.encode(keyWords)
print(keyWords,sim(u,v))

keyWords = " ".join(model.extract_keywords(texts[n],use_maxsum=True,top_n=k))
v = vectorizer.encode(keyWords)
print(keyWords,sim(u,v))

keyWords = " ".join(model.extract_keywords(texts[n],use_mmr=True,diversity=0.2,top_n=k))
v = vectorizer.encode(keyWords)
print(keyWords,sim(u,v))

keyWords = " ".join(model.extract_keywords(texts[n],use_mmr=True,use_maxsum=True, top_n=k))
v = vectorizer.encode(keyWords)
print(keyWords,sim(u,v))