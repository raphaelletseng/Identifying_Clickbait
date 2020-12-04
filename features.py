import numpy as np
import pandas as pd
import scipy 
import nltk
from nltk.tag import pos_tag
import re
from helper import process_text, sim_preprocess, is_contraction

# def loadGloveModel(gloveFile):
#   print('Loading Glove Model')
#   with open(gloveFile, encoding='utf8') as f:
#     content = f.readlines()
#   model = {}
#   for line in content:
#     splitLine = line.split()
#     word = splitLine[0]
#     embedding = np.array([float(val) for val in splitLine[1:]])
#     model[word] = embedding
#   print('Done.',len(model),' words loaded.')
#   return model

# glove_file = 'glove.6B.50d.txt'
# model = loadGloveModel(glove_file)

# def cosine_dist_between_two_words(word1, word2):
#   return (1 - scipy.spatial.disatnce.cosine(model[word1], model[word2]))

# def cosine_dist_wordembedding_method(s1, s2):
#   vec1 = np.mean([model[word] for word in s1], axis=0)
#   vec2 = np.mean([model[word] for word in s2], axis=0)
#   cosine = scipy.spatial.distance.cosine(vec1, vec2)
#   return (1-cosine)*100
  # print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')

def word_count(text):
  return len(str(text).split(' '))

def get_sentences(text):
  return nltk.tokenize.sent_tokenize(text)

# doesn't work
def get_paragraphs(text):
  sent = ''
  sentences = []
  for c in text:
      if c == '\n':
          sentences.append(sent)
          sent = ''
      else:
          sent += c
  if sent != '':
      sentences.append(sent)
  return sentences

def get_proper_nouns(text):
  tagged_sent = pos_tag(text.split())
  # [('word1', 'POS tag'),... ]
  return [word for word,pos in tagged_sent if pos == 'NNP']

def avg_word_length(text):
  words = re.sub("[^a-zA-Z]", " ", text)
  words = words.split()
  length = 0
  for word in words:
    length += len(word)
  return length/len(words)

def len_longest_word(text):
  words = re.sub("[^a-zA-Z]", " ", text)
  words = words.split()
  max_length = 0
  for word in words:
    if len(word) > max_length:
      max_length = len(word)
  return max_length

def qm_count(text):
  count = 0
  for c in text:
    if c == '?':
      count += 1
  return count

def num_contractions(text):
  words = text.split()
  count = 0
  for word in words:
    if "'" in word and is_contraction(word):
      count += 1
  return count


article_data = 'articles1.csv'
data = pd.read_csv(article_data,engine='python',usecols=['index','id','title','content'],nrows=100)
data = data[data['content'].notna()]
data = data[data['title'].notna()]

# string = "can't eat without wasn't too aren't"
# print(num_contractions(string))

for i in range(data.shape[0]):
  print(num_contractions(data.iloc[i]['content']))
# print(qm_count(data.iloc[0]['content']))

# sentences = get_sentences(data.iloc[0]['content'])
# title = data.iloc[0]['title']
# sim_score = 0
# for sent in sentences:
#   sim_score += cosine_dist_wordembedding_method(sim_preprocess(title),sim_preprocess(sent))
# print(sim_score/len(sentences))
# print(cosine_dist_wordembedding_method(sim_preprocess(title),sim_preprocess(data.iloc[0]['content'])))

# print(len(get_proper_nouns(data.iloc[0]['content'])))

# data['processed_content'] = data['content'].apply(process_text)
# print(data.iloc[0]['processed_content'])
