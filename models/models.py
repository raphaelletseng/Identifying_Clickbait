import numpy as np
import pandas as pd
import nltk
import string
from sklearn import metrics
# from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from features import read_features

data, labels = read_features()
labels = [round(label) for label in labels]
train_data = data[:3200]
train_labels = labels[:3200]
test_data = data[3200:]
test_labels = labels[3200:]

#-------------------------
#Logistic Regression
clf1 = LogisticRegression(random_state = 0, solver = 'lbfgs').fit(train_data, train_labels)
clf1.score(train_data, train_labels)
clf1_pred = clf1.predict(test_data)
acc_score1 = metrics.accuracy_score(test_labels, clf1_pred)
print('Acc LR: {}'.format(acc_score1))
'''
#---------------------------
#svm
clf2 = svm.SVC(kernel='linear', random_state = 0, C = 0.2)
clf2.fit(train_data, train_labels)
clf2_pred = clf2.predict(test_data)
# print(clf2_pred)
acc_score2 = metrics.accuracy_score(test_labels, clf2_pred)
print('Acc SVM: {}'.format(acc_score2))
'''
#----------------------------
#Naive bayes
clf3 = MultinomialNB().fit(train_data, train_labels)
clf3_pred = clf3.predict(test_data)
acc_score3 = metrics.accuracy_score(test_labels, clf3_pred)
print('Acc NB: {}'.format(acc_score3))

#----------------------------
#Random Forest
# train_data, train_labels = make_classification(n_samples=1600,random_state = 0, shuffle = False) #n_features = 4, n_samples = 1000
clf4 = RandomForestClassifier(n_estimators=400)
clf4.fit(train_data, train_labels)
clf4_pred = clf4.predict(test_data)
acc_score4 = clf4.score(test_data, test_labels)
#acc_score4 = metrics.accuracy_score(test_labels, clf4_pred)
print('Acc Random Forest: {}'.format(acc_score4))
