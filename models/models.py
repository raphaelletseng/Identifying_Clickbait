import numpy as np
import pandas as pd
import nltk
import string
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import random

from features import read_features

def get_class(num):
    if num < 0.17:
        return 0
    elif num < 0.5:
        return 1
    elif num < 0.83:
        return 2
    else:
        return 3

data, labels = read_features()
order = random.sample(range(len(data)), len(data))
train_idx = order[:14570]
test_idx = order[14570:]
# 2 classes
labels = [round(label) for label in labels]
# 4 classes
labels = [get_class(label) for label in labels]
train_data = []
train_labels = []
test_data = []
test_labels = []
for idx,d in enumerate(data): 
    if idx in train_idx:
        train_data.append(d)
        train_labels.append(labels[idx])
    else:
        test_data.append(d)
        test_labels.append(labels[idx])

#-------------------------
#Logistic Regression
clf1 = LogisticRegression(random_state = 0, solver = 'lbfgs').fit(train_data, train_labels)
clf1.score(train_data, train_labels)
clf1_pred = clf1.predict(test_data)
acc_score1 = metrics.accuracy_score(test_labels, clf1_pred)
print('Acc LR: {}'.format(acc_score1))
f1_score1 = metrics.f1_score(test_labels, clf1_pred, average='micro')
print('F1 LR: {}'.format(f1_score1))
p_score1 = metrics.precision_score(test_labels, clf1_pred, average ='micro')
print('Precision LR: {}'.format(p_score1))

#----------------------------
#Naive bayes
clf3 = MultinomialNB().fit(train_data, train_labels)
clf3_pred = clf3.predict(test_data)
acc_score3 = metrics.accuracy_score(test_labels, clf3_pred)
print('Acc NB: {}'.format(acc_score3))
f1_score3 = metrics.f1_score(test_labels, clf3_pred, average='micro')
print('F1 LR: {}'.format(f1_score3))
p_score3 = metrics.precision_score(test_labels, clf3_pred, average ='micro')
print('Precision LR: {}'.format(p_score3))

#----------------------------
#Random Forest
clf4 = RandomForestClassifier(n_estimators=400)
clf4.fit(train_data, train_labels)
clf4_pred = clf4.predict(test_data)
acc_score4 = clf4.score(test_data, test_labels)
print('Acc Random Forest: {}'.format(acc_score4))
f1_score4 = metrics.f1_score(test_labels, clf4_pred, average='micro')
print('F1 LR: {}'.format(f1_score4))
p_score4 = metrics.precision_score(test_labels, clf4_pred, average ='micro')
print('Precision LR: {}'.format(p_score4))
