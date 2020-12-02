import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import random
from torch.utils.data import TensorDataset, DataLoader

train_data = [np.array(random.uniform(size=(100))) for _ in range(20000)]
train_labels = random.randint(10, size=(20000))

test_data = [np.array(random.uniform(size=(100))) for _ in range(1000)]
test_labels = random.randint(10, size=(1000))

# transform to torch tensor
x_train_tensor = torch.Tensor(train_data) 
y_train_tensor = torch.Tensor(train_labels)

x_test_tensor = torch.Tensor(test_data) 
y_test_tensor = torch.Tensor(test_labels)

train_dataset = TensorDataset(x_train_tensor,y_train_tensor) 
test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

batch_size = 64         # hyper-parameter 
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
    return output

input_size = 100
hidden_size = [2000, 1000]
num_classes = 10

model = MLP(input_size, hidden_size, num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs): 
    loss = 0 
    n_iter = 0 

    for i, (vectors, labels) in enumerate(train_loader): 
        optimizer.zero_grad() 
        outputs = model(vectors)
        loss_bs = criterion(outputs, labels.long())
        loss_bs.backward()
        optimizer.step()
        loss += loss_bs
        n_iter += 1
        # print(loss)

    # print('loss: {}'.format(loss))
    if epoch%1 == 0:
        print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, loss/n_iter))


for i, (vectors, labels) in enumerate(test_loader):
  outputs = model(vectors)
  values, predictions = torch.max(outputs,1) # (max value for each row, col # of max value)
  print(predictions)
  break