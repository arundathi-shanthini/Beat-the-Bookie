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
        self.layer_1 = torch.nn.Linear(72, 36)
        self.layer_2 = torch.nn.Linear(36, 18)
        self.layer_3 = torch.nn.Linear(18, 18)
        self.layer_4 = torch.nn.Linear(18, 9)
        self.layer_5 = torch.nn.Linear(9, 3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        return x

net = NeuralNetworkCalculator()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

training_data=pd.read_csv('epl-training.csv')

win = training_data['FTR']
hometeam = training_data['HomeTeam']
awayteam = training_data['AwayTeam']



HT = torch.tensor(pd.get_dummies(hometeam).values)
AT = torch.tensor(pd.get_dummies(awayteam).values)
W = torch.tensor(pd.get_dummies(win).values)
print(HT)

HAT = torch.transpose(torch.cat((torch.transpose(HT, 0, 1),torch.transpose(AT, 0, 1)),0), 0, 1)
print(HAT)
print(HAT.size())



"""for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(0, x.shape[0]):
        # get the inputs
        inputs, labels = x[i], y[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
print('Finished Training')"""
