import numpy as np
import pandas as pd
import nltk
import string
from sklearn import metrics

train_data = [np.array(random.uniform(size = (100))) for _ in range (20000)]
train_labels = random.randint(10, size = (20000))

test_data = [np.array(random.uniform(size=(100))) for _ in range(1000)]
test_labels = random.randint(10, size=(1000))

#-------------------------
#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(random_state = 0, solver = 'lbfgs').fit(train_data, train_labels)
clf1.score(train_data, train_labels)
clf1_pred = clf1.predict(test_data)
acc_score1 = metrics.accuracy_score(test_labels, clf1_pred)
print('Acc LR: {}'.format(acc_score1))

#---------------------------
#svm
from sklearn import svm
clf2 = svm.SVC(kernel='linear', random_state = 0, C = 0.2)
clf2.fit(train_data, train_labels)
clf2_pred = clf2.predict(test_data)
print(clf2_pred)
acc_score2 = metrics.accuracy_score(test_labels, clf2_pred)
print('Acc SVM: {}'.format(acc_score2))

#----------------------------
#Naive bayes
from sklearn.naive_bayes import MultinomialNB
clf3 = MultinomialNB().fit(train_data, train_labels)
clf3_pred = clf3.predict(test_data)
acc_score3 = metrics.accuracy_score(test_labels, clf3_pred)
print('Acc NB: {}'.format(acc_score3))
