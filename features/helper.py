# for helper functions
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
import json

# contraction dictionary
c_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(c_dict.keys()))

# stop words
# add_stop = ['said', 'say', '...', 'like', 'cnn', 'ad', 'just', 'feel', 'things', 'know', 'think', 'burnaby', 'langley', 'ridge', 'because',
#             'people', 'thing', 'day', 'time', 'need', 'make', 'cent', 'ap'] 
stop_words = ENGLISH_STOP_WORDS

punc = list(set(string.punctuation))

def casual_tokenizer(text):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

def expandContractions(text, c_re=c_re):
    def replace(match):
        return c_dict[match.group(0)]
    return c_re.sub(replace, text)

def sim_preprocess(text):
    letters_only_text = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only_text.lower().split()
    stopword_set = set(stop_words)
    cleaned_words = list(set([w for w in words if w not in stopword_set]))
    return cleaned_words

def process_text(text):
    text = casual_tokenizer(text)
    text = [each.lower() for each in text]
    text = [re.sub('[0-9]+', '', each) for each in text]
    text = [expandContractions(each, c_re=c_re) for each in text]
    text = [w for w in text if w not in stop_words]
    text = [each for each in text if len(each) > 1]
    text = [each for each in text if ' ' not in each]
    text = [SnowballStemmer('english').stem(each) for each in text]
    text = [w for w in text if w not in punc]
    return text

def whitespace_tokenizer(text): 
    pattern = r"(?u)\b\w\w+\b" 
    tokenizer_regex = RegexpTokenizer(pattern)
    tokens = tokenizer_regex.tokenize(text)
    return tokens

def unique_words(text): 
    # Can be replaced by np.unique(list or text)
    ulist = []
    [ulist.append(x) for x in text if x not in ulist]
    return ulist

def is_contraction(word):
    if word in c_dict:
        return True
    else:
        return False

def is_stopword(word):
    if word in stop_words:
        return True
    else:
        return False

def get_sentences(text):
  return nltk.tokenize.sent_tokenize(text)

def porterStem(paragraph):
    """
    Temporary porter stemmer for some debugging.
    Also removes stop words (nltk defined).
    @return stemmed paragraph (str)
    """
    stem = nltk.stem.porter.PorterStemmer().stem
    data = []
    for word in paragraph.split():
        if(word in nltk.corpus.stopwords.words('english')):
            continue
        data.append(stem(word))
        
    return " ".join(data)

def removePunctuation(paragraph):
    """
    Replaces non-ascii characters with ? and then replaces all punctuation with whitespace.
    @return ascii string without punctuation
    """
    punct = re.compile('[%s]' % re.escape(string.punctuation))
    paragraph = paragraph.encode("ascii", errors="replace").decode()
    return punct.sub(' ', paragraph)

def preprocess(paragraph):
    """
    Keeps the numbers, still needs to expand contractions
    Similar to process_text
    """
    return porterStem(removePunctuation(paragraph))

def loadAndProcessJsonData(maxArticles=None):
    """
    Loads the JSON data into three lists and applies the preprocessing 
    in the "preprocess" function.
    Adds \n between paragraphs.
    Only loads maxArticles number of articles, for debugging (if None it loads all)
    @return preprocessed lists titles, texts, labels
    """
    
    texts = []
    titles = []
    labels = []
    dictLabels = {}

    with open('../data/truth.jsonl') as file:
        for line in file.readlines():
            d = json.loads(line)
            dictLabels[d['id']] = d['truthMean']
    
    index=0
    with open('../data/instances.jsonl') as file:
        for line in file.readlines():
            d = json.loads(line)
            texts.append("\n ".join([preprocess(p) for p in d['targetParagraphs']]))

            titles.append(preprocess(d['targetTitle']))
            labels.append(dictLabels[d['id']])
            
            index+=1
            if(maxArticles != None and index >= maxArticles):
                break
    
    return titles,texts,labels    

def loadJsonData(maxArticles=None):
    """Loads the JSON data into three lists.
    Adds \n between paragraphs
    Only loads maxArticles number of articles, for debugging (if None it loads all)
    @return titles, texts, labels
    """
    texts = []
    titles = []
    labels = []
    dictLabels = {}

    with open('../data/truth.jsonl') as file:
        for line in file.readlines():
            d = json.loads(line)
            dictLabels[d['id']] = d['truthMean']
    index=0
    with open('../data/instances.jsonl') as file:
        for line in file.readlines():
            d = json.loads(line)
            texts.append("\n ".join([(p) for p in d['targetParagraphs']]))

            titles.append((d['targetTitle']))
            labels.append(dictLabels[d['id']])
            
            index+=1
            if(maxArticles != None and index >= maxArticles):
                break
    
    return titles,texts,labels    

def getPOSTags(data):
    """
    Takes in a string and returns a list of (word, POS tag) pairs.
    """
    return nltk.pos_tag(casual_tokenizer(data))

'''
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
  print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
'''
