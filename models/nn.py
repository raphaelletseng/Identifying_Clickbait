import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import random
from torch.utils.data import TensorDataset, DataLoader

train_data = np.array([random.uniform(low=0,high=1,size=50) for _ in range(10000)])
train_labels = np.array([np.sum(row) + random.uniform(0,1) for row in train_data])
label_mean = np.mean(train_labels)
label_std = np.std(train_labels)
train_labels = np.array([(val-label_mean)/label_std for val in train_labels])
train_data = torch.tensor(train_data).float()
train_labels = torch.tensor(train_labels).float().unsqueeze(1)

# print(train_data)
# print(train_labels)

test_data = np.array([random.uniform(low=0,high=1,size=50) for _ in range(3000)])
test_labels = np.array([np.sum(row) + random.uniform(0,1) for row in test_data])
test_labels = np.array([(val-label_mean)/label_std for val in test_labels])
test_data = torch.tensor(test_data).float()
test_labels = torch.tensor(test_labels).float().unsqueeze(1)

# print(test_data.shape)
# print(test_labels.shape)

train_dataset = TensorDataset(train_data,train_labels) 
test_dataset = TensorDataset(test_data,test_labels)

batch_size = 64 # hyper-parameter 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(MLP, self).__init__()

    h1, h2 = hidden_size

    self.block = nn.Sequential(
        nn.Linear(input_size, h1), 
        nn.Linear(h1, h2),
        nn.Linear(h2, num_classes)
        )

  def forward(self, x):
    output = self.block(x)
    # output = nn.functional.relu(output)
    return output

input_size = 50
hidden_size = [100, 200]
num_classes = 1

model = MLP(input_size, hidden_size, num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()

num_epochs = 10

for epoch in range(num_epochs): 
    loss = 0 
    n_iter = 0 

    for i, (vectors, labels) in enumerate(train_loader): 
      optimizer.zero_grad()
      outputs = model(vectors)
      loss_bs = criterion(outputs, labels)
      loss_bs.backward()
      optimizer.step()
      loss += loss_bs
      n_iter += 1
    print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, loss/n_iter))


for i, (vectors, labels) in enumerate(test_loader):
  outputs = model(vectors)
  values, predictions = torch.max(outputs,1) # (max value for each row, col # of max value)