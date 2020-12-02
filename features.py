import numpy as np
import pandas as pd
import scipy 
from helper import process_text, word_count

def loadGloveModel(gloveFile):
  print('Loading Glove Model')
  with open(gloveFile, encoding='utf8') as f:
    content = f.readlines()
  model = {}
  for line in content:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
  print('Done.',len(model),' words loaded.')
  return model

glove_file = 'glove.6B.50d.txt'
model = loadGloveModel(glove_file)

def cosine_dist_between_two_words(word1, word2):
  return (1 - scipy.spatial.disatnce.cosine(model[word1], model[word2]))

def cosine_dist_wordembedding_method(s1, s2):
  vec1 = np.mean([model[word] for word in s1], axis=0)
  vec2 = np.mean([model[word] for word in s2], axis=0)
  cosine = scipy.spatial.distance.cosine(vec1, vec2)
  return (1-cosine)*100
#   print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')

article_data = 'articles1.csv'
data = pd.read_csv(article_data,engine='python',usecols=['index','id','title','content'],nrows=300)
data = data[data['content'].notna()]
data = data[data['title'].notna()]

print(data.iloc[0]['content'])

# wordcounts = []
# for idx, row in data.iterrows():
#     wordcounts.append(word_count(row['content']))

# data['processed_content'] = data['content'].apply(process_text)
# print(data.head())
