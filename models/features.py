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
from multiprocessing.dummy import Pool as ThreadPool
import threading

from helper import process_text, sim_preprocess, is_contraction, is_stopword, loadAndProcessJsonData, getPOSTags, removePunctuation, get_sentences

def token_count(text):
  tokens = FreqDist(text)
  return len(tokens)

# DISCUSS AS A CHALLENGE IN FINAL PAPER
# returns list of proper nouns (content)
# this does not work well because it thinks too many things are NNP, NNPS if they are capitalized
# also because it does not capture 2- or 3-gram NNPs
# def get_proper_nouns(text):
#   tagged_sent = getPOSTags([text])
#   # [('word1', 'POS tag'),... ]
#   print(tagged_sent)
#   return [word for (word,pos) in tagged_sent[0] if pos == 'NNP' or pos == 'NNPS']

def word_lengths(text,text_length):
  length = 0
  max_length = 0
  for word in text:
    length += len(word)
    if len(word) >= max_length:
      max_length = len(word)
  return length/text_length, max_length

def counts(text,content_flag):
  qm_count = 0
  em_count = 0
  digit_count = 0
  if content_flag:
    for c in text:
      if c == '?':
        qm_count += 1
      elif c == '!':
        em_count += 1
    return qm_count, em_count
  else:
    for c in text:
      if c == '?':
        qm_count = 1
      elif c == '!':
        em_count = 1
      elif c.isdigit():
        digit_count = 1
    return qm_count, em_count, digit_count

def num_contractions(text):
  count = 0
  for word in text:
    if "'" in word and is_contraction(word):
      count += 1
  return count

# returns True if the first word is an adverb (headline)
def contains_adverb(POS_tags):
  if(POS_tags == None or len(POS_tags) == 0 or len(POS_tags[0]) == 0):
    return False
  if 'RB' in POS_tags[0][1]:
    return True
  return False

# returns number of superlative adjectives, superlative adverbs (content)
def superlative_adj_adv_count(POS_tags):
  adj_count = 1
  adv_count = 0
  for _, tag in POS_tags:
    if tag == 'JJS':
      adj_count += 1
    elif tag == 'RBS':
      adv_count += 1
  return adj_count, adv_count

def count_stopwords(text):
  count = 0
  for word in text:
    if is_stopword(word):
      count += 1
  return count

def posTagFeatures(taggedArticle):   
    """
    Takes in a list of (word, tag) pairs and returns various metrics based on them.
    @return a list of the results for each of the desired features (mostly some . The list is 
    [NNP,NNP_NNP,IN,NNP_VBZ,IN_NNP,WRB,NN,QM>0,PRP,VBZ,NNP_NNP_VBZ,NN_IN,NN_IN_NNP,
    NNP_any,PRP_VBP,WP,DT,NNP_IN,IN_NNP_NNP,POS,IN_NN,NNP_NNS,IN_JJ,NNP_POS,WDT,
    NN_NN,NN_NNP,NNP_VBD,RB,NNP_NNP_NNP,NNP_NNP_NN,RBS,VBN,VBN_IN,NUMBER_NP_VB>0,
    JJ_NNP,NNP_NN_NN,DT_NN,EX>0]
    """
    NNP=0
    NNP_NNP=0
    IN=0
    NNP_VBZ=0
    IN_NNP=0
    WRB=0
    NN=0
    #Exists
    QM=0
    PRP=0
    VBZ=0
    NNP_NNP_VBZ=0
    NN_IN=0
    NN_IN_NNP=0
    #Assumed the 27th feature, POS 2-gram NNP . meant any case where there are two tags and the previous is NNP
    #It could be that the NNP was the last in the sentence though
    NNP_any=0
    PRP_VBP=0
    WP=0
    DT=0
    NNP_IN=0
    IN_NNP_NNP=0
    POS=0
    IN_NN=0
    NNP_NNS=0
    IN_JJ=0
    NNP_POS=0
    WDT=0
    NN_NN=0
    NN_NNP=0
    NNP_VBD=0
    RB=0
    NNP_NNP_NNP=0
    NNP_NNP_NN=0
    RBS=0
    VBN=0
    VBN_IN=0
    #Exists
    NUMBER_NP_VB=0
    JJ_NNP=0
    NNP_NN_NN=0
    DT_NN=0
    #Exists
    EX=0
    
    prevTag=None
    prevPrevTag=None
    
    for word,tag in taggedArticle:
        if(tag == "NNP"):
            NNP+=1
            if(prevTag == "NNP"):
                NNP_NNP+=1
                if(prevPrevTag == "IN"):
                    IN_NNP_NNP+=1
            elif(prevTag == "IN"):
                IN_NNP+=1
                if(prevPrevTag == "NN"):
                    NN_IN_NNP+=1
        elif(tag == "IN"):
            IN+=1
            if(prevTag == "NN"):
                NN_IN+=1
            elif(prevTag == "NNP"):
                NNP_IN+=1
        elif(tag == "WRB"):
            WRB+=1
        elif(tag == "NN"):
            NN+=1
            if(prevTag == "IN"):
                IN_NN+=1
        elif(tag == "QM"):
            QM+=1
        elif(tag == "PRP"):
            PRP+=1
        elif(tag == "VBZ"):
            VBZ+=1
            if(prevTag == "NNP"):
                NNP_VBZ+=1
                if(prevPrevTag == "NNP"):
                    NNP_NNP_VBZ+=1
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
            
        #The above format is easy to make a mistake, so the following changes that (but it's less efficient).
        if(tag == "VBP" and prevTag == "PRP"):
            PRP_VBP+=1
        if(tag == "NNS" and prevTag == "NNP"):
            NNP_NNS+=1
        if(tag == "JJ" and prevTag == "IN"):
            IN_JJ+=1
        if(tag == "POS" and prevTag == "NNP"):
            NNP_POS+=1
        if(tag == "NN" and prevTag == "NN"):
            NN_NN+=1
        if(tag == "NNP" and prevTag == "NN"):
            NN_NNP+=1
        if(tag == "VBD" and prevTag == "NNP"):
            NNP_POS+=1
        if(tag == "NNP" and prevTag == "NNP" and prevPrevTag == "NNP"):
            NNP_NNP_NNP+=1
        if(tag == "NN" and prevTag == "NNP" and prevPrevTag == "NNP"):
            NNP_NNP_NN+=1
        if(tag == "IN" and prevTag == "VBN"):
            VBN_IN+=1
        if(tag == "VB" and prevTag == "NP" and prevPrevTag == "NUMBER"):
            NUMBER_NP_VB+=1
        if(tag == "NNP" and prevTag == "JJ"):
            JJ_NNP+=1
        if(tag == "NN" and prevTag == "NN" and prevPrevTag == "NNP"):
            NNP_NN_NN+=1
        if(tag == "NN" and prevTag == "DT"):
            DT_NN+=1
            
        if(prevTag == "NNP"):
            NNP_any+=1
        
        #TODO: check that the tokenizer returns these, and then we need to ignore other punctuation so that the previous tags don't become none or punctuation
        if(word in ['.','!','?']):
            prevTag=None
            prevPrevTag=None
        else:
            prevPrevTag=prevTag
            prevTag = tag
        
    return [NNP,NNP_NNP,IN,NNP_VBZ,IN_NNP,WRB,NN,QM>0,PRP,VBZ,NNP_NNP_VBZ,NN_IN,NN_IN_NNP,NNP_any,PRP_VBP,WP,DT,NNP_IN,IN_NNP_NNP,POS,IN_NN,NNP_NNS,IN_JJ,NNP_POS,WDT,NN_NN,NN_NNP,NNP_VBD,RB,NNP_NNP_NNP,NNP_NNP_NN,RBS,VBN,VBN_IN,NUMBER_NP_VB>0,JJ_NNP,NNP_NN_NN,DT_NN,EX>0]

# get keywords (content)
model = KeyBERT('distilbert-base-nli-mean-tokens')
def get_key_words(text):
  stem = PorterStemmer().stem
  text = [stem(word) for word in text if not is_stopword(word)]
  text = ' '.join(text)
  top_n = 8
  # return [model.extract_keywords(text,top_n=top_n),model.extract_keywords(text,top_n=top_n,use_maxsum=True),model.extract_keywords(text,top_n=top_n,use_mmr=True,diversity=0.2),model.extract_keywords(text,use_mmr=True,use_maxsum=True,top_n=top_n)]
  return model.extract_keywords(text,top_n=top_n)

# get similarity between two strings
vectorizer = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
def get_similarity(text_keywords,title):
  keyword_vec = vectorizer.encode(text_keywords) # text_keywords should be a string, not a list
  title_vec = vectorizer.encode(title)
  return scipy.spatial.distance.cosine(keyword_vec,title_vec)

# checks if first word is who/what/where/why/how/when (title)
def starts_with_q_word(text):
  first_word = text[0].lower()
  if first_word == 'what' or first_word == 'who' or first_word == 'where' or first_word == 'why' or first_word == 'how' or first_word == 'when':
    return True
  else:
    return False

def get_feature_vector(feature_data):
  headline, content = feature_data
  print(headline)
  flat_content = ' '.join(content)
  content_sentences = [get_sentences(paragraph) for paragraph in content]

  qm_count_headline, em_count_headline, digit_count = counts(headline,0)
  qm_count_content, em_count_content = counts(flat_content,1)

  headline = headline.split()
  content = [paragraph.split() for paragraph in content]

  num_contractions_headline = num_contractions(headline)
  num_contractions_content = sum([num_contractions(paragraph) for paragraph in content])
  num_stopwords = sum([count_stopwords(paragraph) for paragraph in content])
  # print(content[0])

  headline = [removePunctuation(word) for word in headline]
  # content = [[word.replace('\'','') for word in paragraph] for paragraph  in content]
  content = [[removePunctuation(word) for word in paragraph] for paragraph in content]
  
  content_words = [word.split()[0] for paragraph in content for word in paragraph if word.split() != []]

  num_words = len(content_words)   
  num_tokens = token_count(' '.join(content_words))
  avg_length_words, longest_word = word_lengths(content_words,num_words)
  avg_length_sentences = num_words/len(content_sentences)
  starts_with_question_word = int(starts_with_q_word(headline))

  stopword_ratio = num_stopwords/num_words
  contraction_ratio = num_contractions_content/num_words

  headline = ' '.join(headline)
  POS_tags_headline = getPOSTags(headline)
  POS_tags_content = getPOSTags(' '.join(content_words))

  adverb = int(contains_adverb(POS_tags_headline))
  super_adj_count, super_adv_count = superlative_adj_adv_count(POS_tags_content)

  POS_counts =  posTagFeatures(POS_tags_content)

  BERT_keywords = ' '.join(get_key_words(content_words))
  document_sim = get_similarity(BERT_keywords,headline)
  sentence_sims = []
  paragraph_sims = []
  for paragraph in content:
    for sent in paragraph:
      sentence_sims.append(get_similarity(BERT_keywords,' '.join(sent)))
    paragraph_sims.append(get_similarity(BERT_keywords,' '.join([' '.join(sent) for sent in paragraph])))
  sentence_sims = sum(sentence_sims)/len(sentence_sims)
  paragraph_sims = sum(paragraph_sims)/len(paragraph_sims)
  
  vector = [qm_count_headline, em_count_headline, digit_count, qm_count_content, em_count_content,
            num_contractions_headline, num_contractions_content, num_stopwords, num_words, num_tokens,
            avg_length_words, longest_word, avg_length_sentences, starts_with_question_word, stopword_ratio,
            contraction_ratio, adverb, super_adj_count, super_adv_count, document_sim, sentence_sims,
            paragraph_sims]
  for item in POS_counts:
    vector.append(int(item))
  print(headline,' DONE')
  return (feature_data[0],vector)

# article_data = 'articles1.csv'
# data = pd.read_csv(article_data,engine='python',usecols=['index','id','title','content'],nrows=100,encoding='unicode_escape')
# data = data[data['content'].notna()]
# data = data[data['title'].notna()]

def get_data():
  titles, texts, labels = loadAndProcessJsonData()
  spreadsheet = [(titles[idx],texts[idx]) for idx in range(len(titles))]
  pool = ThreadPool(8)
  results = pool.map(get_feature_vector, spreadsheet)
  # print(results)
  pool.close()
  pool.join()
  # f = open('data-3c.txt', 'w')
  f = open('data-3d.txt','w')
  f.write('results: {} labels: {}\n'.format(results,labels))
  f.close()

  # tagged = getPOSTags(texts)
  # print(posTagFeatures(tagged[0]))
<<<<<<< HEAD
=======
  # f = open('feature_vectors.txt','w')
  # for idx in range(len(titles)):
  #   print('getting feature vector {}...'.format(idx))
  #   vector = get_feature_vector(titles[idx],texts[idx])
  #   print('writing feature vector {}...'.format(idx))
  #   f.write('label: {}, vector: {}\n'.format(labels[idx],vector))
  # f.close()

def read_features():
  vectors = []
  labels = []
  
  starting_flag = 0
  f = open('data/data-3a.txt')
  data = f.read()
>>>>>>> a47e3c81459bc4476dde4725ae5e521054143791
  f.close()
  l = len(data)
  i = 0
  c = data[i]
  while(i<l):
    if c == ':':
      starting_flag += 1
      i += 1
      c = data[i]
    # outer bracket (results only)
    elif c == '[' and starting_flag == 1:
      starting_flag += 1
      i += 1
      c = data[i]
    elif c == '[' and starting_flag == 3:
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        labels.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      break
    # inner bracket 
    elif c == '[':
      vector = []
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        vector.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      vectors.append(vector)
      i += 1
      c = data[i]
    else:
      i += 1
      c = data[i]
  
  starting_flag = 0
  f = open('data/data-3b.txt')
  data = f.read()
  f.close()
  l = len(data)
  i = 0
  c = data[i]
  while(i<l):
    if c == ':':
      starting_flag += 1
      i += 1
      c = data[i]
    # outer bracket (results only)
    elif c == '[' and starting_flag == 1:
      starting_flag += 1
      i += 1
      c = data[i]
    elif c == '[' and starting_flag == 3:
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        labels.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      break
    # inner bracket 
    elif c == '[':
      vector = []
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        vector.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      vectors.append(vector)
      i += 1
      c = data[i]
    else:
      i += 1
      c = data[i]

  starting_flag = 0
  f = open('data/data-3c.txt')
  data = f.read()
  f.close()
  l = len(data)
  i = 0
  c = data[i]
  while(i<l):
    if c == ':':
      starting_flag += 1
      i += 1
      c = data[i]
    # outer bracket (results only)
    elif c == '[' and starting_flag == 1:
      starting_flag += 1
      i += 1
      c = data[i]
    elif c == '[' and starting_flag == 3:
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        labels.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      break
    elif c == '(':
      vector = []
      i += 2
      c = data[i]
      while(c != '[' or not data[i+1].isdigit()):
        i += 1
        c = data[i]
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        vector.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      vectors.append(vector)
      i += 2
      c = data[i]
    else:
      i += 1
      c = data[i]

  starting_flag = 0
  f = open('data/data-3d.txt')
  data = f.read()
  f.close()
  l = len(data)
  i = 0
  c = data[i]
  while(i<l):
    if c == ':':
      starting_flag += 1
      i += 1
      c = data[i]
    # outer bracket (results only)
    elif c == '[' and starting_flag == 1:
      starting_flag += 1
      i += 1
      c = data[i]
    elif c == '[' and starting_flag == 3:
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        labels.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      break
    elif c == '(':
      vector = []
      i += 2
      c = data[i]
      while(c != '[' or not data[i+1].isdigit()):
        i += 1
        c = data[i]
      i += 1
      c = data[i]
      while(c != ']'):
        if c == ' ':
          i += 1
          c = data[i]
          continue
        num = ''
        while(c != ',' and c != ']'):
          num += str(c)
          i += 1
          c = data[i]
        vector.append(float(num))
        if c == ',':
          i += 1
          c = data[i]
      vectors.append(vector)
      i += 2
      c = data[i]
    else:
      i += 1
      c = data[i]
  
  f = open('feature_vectors.txt')
  data = f.readlines()
  f.close()
  
  for line in data:
    i = 0
    c = line[i]
    while(not c.isdigit()):
      i += 1
      c = line[i]
    num = ''
    while(c != ','):
      num += str(c)
      i += 1
      c = line[i]
    labels.append(float(num))
    while(c != '['):
      i += 1
      c = line[i]
    i += 1
    c = line[i]
    vector = []
    while(c != ']'):
      if c == ' ':
        i += 1
        c = line[i]
        continue
      num = ''
      while(c != ',' and c != ']'):
        num += str(c)
        i += 1
        c = line[i]
      vector.append(float(num))
      if c == ',':
        i += 1
        c = line[i]
    vectors.append(vector)

  # print(len(vectors))
  # print(len(labels))   
  return vectors,labels

# get_data()
read_features()
