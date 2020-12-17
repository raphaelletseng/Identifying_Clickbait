import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
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

# 80:20 train:test
train_idx = order[:14570]
test_idx = order[14570:]
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

train_labels_reg = torch.tensor(train_labels)
test_labels_reg = torch.tensor(test_labels)

# 2 classes
train_labels = torch.tensor([round(label) for label in train_labels])
test_labels = torch.tensor([round(label) for label in test_labels])
# 4 classes
train_labels = torch.tensor([get_class(label) for label in train_labels])
test_labels = torch.tensor([get_class(label) for label in test_labels])

count_rows = list(range(22,60))
for i in range(9):
  count_rows.append(i)
count_rows.append(17)
count_rows.append(18)

train_dataT = np.matrix(train_data)
train_dataT = train_dataT.transpose()
means = []
stds = []
for idx, row in enumerate(train_data):
  if idx in count_rows:
    means.append(np.mean(row))
    stds.append(np.std(row))
  else:
    means.append(0)
    stds.append(0)

normalized_train_data = []
for d in train_data:
  new_d = d
  for i in count_rows:
    new_d[i] = (d[i]-means[i])/stds[i]
  normalized_train_data.append(new_d)

normalized_test_data = []
for d in test_data:
  new_d = d
  for i in count_rows:
    new_d[i] = (d[i]-means[i])/stds[i]
  normalized_test_data.append(new_d)

# classification
train_dataset = TensorDataset(torch.tensor(normalized_train_data).float(),train_labels) 
test_dataset = TensorDataset(torch.tensor(normalized_test_data).float(),test_labels)

# regression
train_dataset_reg = TensorDataset(torch.tensor(normalized_train_data).float(),train_labels_reg.float())
test_dataset_reg = TensorDataset(torch.tensor(normalized_test_data).float(),test_labels_reg.float())

# 2-class classification and regression 
batch_size = 64 
# 4-class classification
batch_size = 16

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

train_loader_reg = torch.utils.data.DataLoader(dataset=train_dataset_reg,batch_size=batch_size,shuffle=True)
# test_loader_reg = torch.utils.data.DataLoader(dataset=test_dataset_reg,batch_size=batch_size,shuffle=True)

# classification
class MLP_clf(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(MLP_clf, self).__init__()
    h1, h2 = hidden_size
    self.block = nn.Sequential(
        nn.Linear(input_size, h1), 
        nn.Linear(h1, h2),
        nn.Linear(h2, num_classes))

  def forward(self, x):
    output = self.block(x)
    return output

# regression
class MLP_reg(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(MLP_reg, self).__init__()
    h1, h2 = hidden_size
    self.block = nn.Sequential(
        nn.Linear(input_size, h1), 
        nn.Linear(h1, h2),
        nn.Linear(h2, num_classes))

  def forward(self, x):
    output = self.block(x)
    return output

# train classifciation
# 2 classes
clf = MLP_clf(input_size=61, hidden_size=(100,50), num_classes=2)
# 4 classes
clf = MLP_clf(input_size=61, hidden_size=(100,50), num_classes=4)
optimizer_clf = optim.SGD(clf.parameters(),lr=0.01,momentum=0.9)
criterion = nn.CrossEntropyLoss()

num_epochs = 20
for epoch in range(num_epochs): 
    loss = 0 
    n_iter = 0 

    for i, (vectors, labels) in enumerate(train_loader): 
      optimizer_clf.zero_grad()
      outputs = clf(vectors)
      loss_bs = criterion(outputs, labels)
      loss_bs.backward()
      optimizer_clf.step()
      loss += loss_bs
      n_iter += 1

    print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, loss/n_iter))

correct = []
f1s = []
precisions = []
losses = []

# test classicification
outputs = clf(torch.tensor(normalized_test_data).float())
values, predictions = torch.max(outputs,1)
loss = criterion(outputs,test_labels.long())
print('acc: ',metrics.accuracy_score(test_labels, predictions))
print('f1: ',metrics.f1_score(test_labels,predictions,average='micro',zero_division=1))
print('precision: ',metrics.precision_score(test_labels,predictions,average='micro',zero_division=1))
print('loss: ',loss.item())

# test regression
reg = MLP_reg(input_size=61,hidden_size=(100,50),num_classes=1)
optimizer_reg = optim.SGD(reg.parameters(),lr=0.001,momentum=0.9)

num_epochs = 15
criterion = nn.MSELoss()
for epoch in range(num_epochs): 
    loss = 0 
    n_iter = 0 

    for i, (vectors, labels) in enumerate(train_loader_reg): 
      optimizer_reg.zero_grad()
      outputs = reg(vectors)
      loss_bs = criterion(outputs,labels.unsqueeze(0))
      loss_bs.backward()
      optimizer_reg.step()
      loss += loss_bs
      n_iter += 1
  
    print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, loss/n_iter))

# test regression
outputs = clf(torch.tensor(normalized_test_data).float())
values, predictions = torch.max(outputs,1)
loss = criterion(outputs,test_labels_reg)
print('loss: ',loss.item())
