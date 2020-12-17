import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import random
from torch.utils.data import TensorDataset, DataLoader

from features import read_features

data, labels = read_features()
labels_clf = [round(label) for label in labels]
train_data = data[:1600]
train_labels = torch.tensor(labels_clf[:1600])
test_data = data[1600:]
test_labels = torch.tensor(labels_clf[1600:])

train_labels_reg = torch.tensor(labels[:1600])
test_labels_reg = torch.tensor(labels[1600:])

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

train_dataset = TensorDataset(torch.tensor(normalized_train_data).float(),train_labels) 
test_dataset = TensorDataset(torch.tensor(normalized_test_data).float(),test_labels)

train_dataset_reg = TensorDataset(torch.tensor(normalized_train_data).float(),train_labels_reg.float())
test_dataset_reg = TensorDataset(torch.tensor(normalized_test_data).float(),test_labels_reg.float())

batch_size = 64 # hyper-parameter 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

train_loader_reg = torch.utils.data.DataLoader(dataset=train_dataset_reg,batch_size=batch_size,shuffle=True)
test_loader_reg = torch.utils.data.DataLoader(dataset=test_dataset_reg,batch_size=batch_size,shuffle=True)

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

clf = MLP_clf(input_size=61, hidden_size=(100,50), num_classes=2)
reg = MLP_reg(input_size=61,hidden_size=(100,50),num_classes=1)
optimizer_clf = optim.SGD(clf.parameters(),lr=0.01,momentum=0.9)
optimizer_reg = optim.SGD(reg.parameters(),lr=0.001,momentum=0.9)
criterion = nn.CrossEntropyLoss()

num_epochs = 15
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
for i, (vectors, labels) in enumerate(test_loader):
  outputs = clf(vectors)
  values, predictions = torch.max(outputs,1) # (max value for each row, col # of max value)
  count = 0
  for idx in range(len(predictions)):
    if predictions[idx] == labels[idx]:
      count += 1
  correct.append(count/len(predictions))

print('classification: {}'.format(np.mean(correct)))


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

losses = []
for i, (vectors, labels) in enumerate(test_loader):
  outputs = reg(vectors)
  loss = criterion(outputs,labels)
  losses.append(loss.item())

print('regression loss: {}'.format(np.mean(losses)))
