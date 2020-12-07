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

# i will organize this at some point so we don't call so many functions over and over again when we're calculating features - miya

def word_count(text):
  return len(str(text).split(' '))

def get_sentences(text):
  return nltk.tokenize.sent_tokenize(text)

# returns list of proper nouns (headline, content)
def get_proper_nouns(text):
  tagged_sent = pos_tag(text.split())
  # [('word1', 'POS tag'),... ]
  return [word for word,pos in tagged_sent if pos == 'NNP']

# returns average length of word (content)
def avg_word_length(text):
  words = re.sub("[^a-zA-Z]", " ", text)
  words = words.split()
  length = 0
  for word in words:
    length += len(word)
  return length/len(words)

# returns length of longest word (headline, content)
def len_longest_word(text):
  words = re.sub("[^a-zA-Z]", " ", text)
  words = words.split()
  max_length = 0
  for word in words:
    if len(word) > max_length:
      max_length = len(word)
  return max_length

# returns True if there is a question mark (headline)
def qm_count(text):
  for c in text:
    if c == '?':
      return True
  return False

# returns True if there is an exclamation mark (headline)
def em_count(text):
  for c in text:
    if c == '!':
      return True
  return False

# returns number of contractions (headline, content)
def num_contractions(text):
  words = text.split()
  count = 0
  for word in words:
    if "'" in word and is_contraction(word):
      count += 1
  return count

# returns True if there is a digit (headline)
def contains_number(text):
  return any(c.isdigit() for c in text)

# returns True if the first word is an adverb (headline)
def contains_adverb(text):
  tagged_sent = pos_tag(text.split())
  if 'RB' in tagged_sent[0][1]:
    return True
  return False

# returns average number of words per sentence (content)
def avg_words_per_sentence(text):
  return word_count(text)/len(get_sentences(text))

article_data = 'articles1.csv'
data = pd.read_csv(article_data,engine='python',usecols=['index','id','title','content'],nrows=100)
data = data[data['content'].notna()]
data = data[data['title'].notna()]


# data['processed_content'] = data['content'].apply(process_text)
# print(data.iloc[0]['processed_content'])
