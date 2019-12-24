import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class NeuralNetworkCalculator(nn.Module):
    def __init__(self):
        super(NeuralNetworkCalculator, self).__init__()
        self.layer_1 = torch.nn.Linear(10, 8)
        self.layer_2 = torch.nn.Linear(8, 8)
        self.layer_3 = torch.nn.Linear(8, 6)
        self.layer_4 = torch.nn.Linear(6, 3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        return x

net = NeuralNetworkCalculator()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.00001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

training_data=pd.read_csv('epl-training.csv')


total_step = 4180

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(0, HAT.shape[0]):
        # get the inputs
        inputs, labels = HAT[i], W[i]
        """mu, sigma = 0, 0.01
        noise = torch.from_numpy(np.random.normal(mu, sigma, [1,72])).float()
        inputs = inputs + noise"""
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            running_loss = 0.0
print('Finished Training')
