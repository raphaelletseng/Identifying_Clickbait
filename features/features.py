import numpy as np
import pandas as pd
import scipy 
import nltk
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from helper import process_text, sim_preprocess, is_contraction, is_stopword, loadAndProcessJsonData, getPOSTags, removePunctuation

# i will organize this at some point so we don't call so many functions over and over again when we're calculating features - miya

def word_count(text):
  return len(str(text).split(' '))

def get_sentences(text):
  return nltk.tokenize.sent_tokenize(text)

def token_count(text):
  text = text.split()
  text = [removePunctuation(word) for word in text]
  tokens = FreqDist(text)
  return len(tokens)

# returns list of proper nouns (headline, content)
# this does not work well because it thinks too many things are NNP, NNPS if they are capitalized
# also because it does not capture 2- or 3-gram NNPs
def get_proper_nouns(text):
  tagged_sent = getPOSTags([text])
  # [('word1', 'POS tag'),... ]
  print(tagged_sent)
  return [word for (word,pos) in tagged_sent[0] if pos == 'NNP' or pos == 'NNPS']

sample = 'Capitalize every word, Mrs. Robinson'
print(get_proper_nouns(sample))

# returns average length of word (content)
def avg_word_length(text):
  words = removePunctuation(text)
  words = words.split()
  length = 0
  for word in words:
    length += len(word)
  return length/len(words)

# returns length of longest word (headline, content)
def len_longest_word(text):
  words = removePunctuation(text)
  words = words.split()
  max_length = 0
  for word in words:
    if len(word) > max_length:
      max_length = len(word)
  return max_length


# returns True if there is a question mark/number of question marks (headline, content)
def qm_count(text,content_flag):
  if content_flag:
    count = 0
    for c in text:
      if c == '?':
        count += 1
    return count
  else:
    for c in text:
      if c == '?':
        return True
    return False

# returns True if there is an exclamation mark/number of exclamation marks (headline, content)
def em_count(text,content_flag):
  if content_flag:
    count = 0
    for c in text:
      if c == '!':
        count += 1
    return count
  else:
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

# returns number of superlative adjectives, superlative adverbs (content)
def superlative_adj_adv_count(text):
  tagged = pos_tag(text.split())
  adj_count = 1
  adv_count = 0
  for _, tag in tagged:
    if tag == 'JJS':
      adj_count += 1
    elif tag == 'RBS':
      adv_count += 1
  return adj_count, adv_count

# returns ratio of stopwords:all words (content)
def stopword_ratio(text):
  text = text.split()
  count = 0
  for word in text:
    if is_stopword(word):
      count += 1
  return count/len(text)

# returns ratio of contractions:all words (content)
def contraction_ratio(text):
  text = text.split()
  count = 0
  for word in text:
    if is_contraction(word):
      count += 1
  return count/len(text)

def posTagFeatures(taggedArticle):   
    """
    Takes in a list of (word, tag) pairs and returns various metrics based on them.
    @return NNP,IN,WRB,NN,QM>0,PRP,VBZ,WP,DT,POS,WDT,RB,RBS,VBN,EX>0
    """
    NNP=0
    #NNP_NNP=0
    IN=0
    #NNP_VBZ=0
    #IN_NNP=0
    WRB=0
    NN=0
    QM=0
    PRP=0
    VBZ=0
    #NNP_NNP_VBZ=0
    #NN_IN=0
    #NN_IN_NNP=0
    #PRP_VBP=0
    WP=0
    DT=0
    POS=0
    WDT=0
    RB=0
    RBS=0
    VBN=0
    EX=0
    
    prevTag=None
    prevPrevTag=None
    
    for word,tag in taggedArticle:
        if(tag == "NNP"):
            NNP+=1
        elif(tag == "IN"):
            IN+=1
        elif(tag == "WRB"):
            WRB+=1
        elif(tag == "NN"):
            NN+=1
        elif(tag == "QM"):
            QM+=1
        elif(tag == "PRP"):
            PRP+=1
        elif(tag == "VBZ"):
            VBZ+=1
        elif(tag == "WP"):
            WP+=1
        elif(tag == "DT"):
            DT+=1
        elif(tag == "POS"):
            POS+=1
        elif(tag == "WDT"):
            WDT+=1
        elif(tag == "RB"):
            RB+=1
        elif(tag == "RBS"):
            RBS+=1
        elif(tag == "VBN"):
            VBN+=1
        elif(tag == "EX"):
            EX+=1
        
        prevPrevTag=prevTag
        prevTag = tag
        
    return NNP,IN,WRB,NN,QM>0,PRP,VBZ,WP,DT,POS,WDT,RB,RBS,VBN,EX>0

# get keywords (content)
model = KeyBERT('distilbert-base-nli-mean-tokens')
def get_key_words(text):
  stem = PorterStemmer().stem
  text = text.split()
  text = [removePunctuation(word) for word in text]
  stemmed_text = []
  for word in text:
    if not is_stopword(word):
      stemmed_text.append(stem(word))
  stemmed_text = ' '.join(stemmed_text)
  top_n = 8
  return model.extract_keywords(stemmed_text,top_n=top_n),model.extract_keywords(stemmed_text,top_n=top_n,use_maxsum=True),model.extract_keywords(stemmed_text,top_n=top_n,use_mmr=True,diversity=0.2),model.extract_keywords(stemmed_text,use_mmr=True,use_maxsum=True,top_n=top_n)

# get similarity between two strings
vectorizer = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
def get_similarity(text_keywords,title):
  keyword_vec = vectorizer.encode(text_keywords) # text_keywords should be a string, not a list
  title_vec = vectorizer.encode(title)
  return scipy.spatial.distance.cosine(keyword_vec,title_vec)

# checks if first word is who/what/where/why/how (title)
def starts_with_q_word(text):
  text = text.split()
  first_word = text[0].lower()
  if first_word == 'what' or first_word == 'who' or first_word == 'where' or first_word == 'why' or first_word == 'how':
    return True
  else:
    return False


article_data = 'articles1.csv'
data = pd.read_csv(article_data,engine='python',usecols=['index','id','title','content'],nrows=100)
data = data[data['content'].notna()]
data = data[data['title'].notna()]
# print(data.iloc[0]['title'])
# print(get_proper_nouns(data.iloc[0]['title']))

# print(get_similarity(data.iloc[0]['content'],data.iloc[0]['title']))
# results = get_key_words(data.iloc[0]['content'])
# for result in results:
#   print(result)

# data['processed_content'] = data['content'].apply(process_text)
# print(data.iloc[0]['processed_content'])

# titles, texts, labels = loadAndProcessJsonData(10)
# tagged = getPOSTags(texts)
# print(posTagFeatures(tagged[0]))