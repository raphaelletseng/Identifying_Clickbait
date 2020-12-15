import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from numpy import random

# torch.manual_seed(1)
# x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
# y = x.pow(2) + 0.2*torch.rand(x.size())
# x,y = Variable(x), Variable(y)
# print(x.shape)
# print(y.shape)

x = np.array([random.uniform(0,1,50) for _ in range(10000)])
y = np.array([np.sum(row) + random.uniform(0,1) for row in x])
x = torch.tensor(x).float()
y = torch.tensor(y).float().unsqueeze(1)

print(x.shape)
print(y.shape)

# x_test = torch.unsqueeze(torch.linspace(-1,1,50),dim=1)
# y_test = x_test.pow(2) + 0.2*torch.rand(x_test.size())

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=50,n_hidden=10,n_output=1)
optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.CrossEntropyLoss()

for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%10 == 0:
        print('Epoch: {}/{}, Loss: {:.4f}'.format(t+1, 200, loss))
    
# outputs = net(x_test)
# values, predictions = torch.max(outputs,1) # (max value for each row, col # of max value)
# print(values)
# print(predictions)
# for i,val in enumerate(values):
#     print('val: {}, label: {}'.format(val,y_test[i]))