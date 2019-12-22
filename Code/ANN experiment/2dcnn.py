import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

num_epochs = 10

class NeuralNetworkCalculator(nn.Module):
    def __init__(self):
        super(NeuralNetworkCalculator, self).__init__()
        self.layer_1 = torch.nn.Conv2d(1, 1, (4,4), stride=(1,1), padding=(0,0))
        self.layer_2 = torch.nn.Conv2d(1, 1, (3,3), stride=(1,1), padding=(1,1))
        self.layer_3 = torch.nn.Conv2d(1, 1, (4,4), stride=(1,1), padding=(0,0))
        self.layer_4 = torch.nn.Linear(264, 3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = x.view(-1, 264)
        x = F.relu(self.layer_4(x))
        return x

net = NeuralNetworkCalculator()

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr = 0.000001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

training_data=pd.read_csv('epl-training.csv')

win = training_data['FTR']
hometeam = training_data['HomeTeam']
awayteam = training_data['AwayTeam']
month = pd.to_datetime(training_data['Date'],format='%d/%m/%Y').dt.month



HT = torch.tensor(pd.get_dummies(hometeam).values)
AT = torch.tensor(pd.get_dummies(awayteam).values)
M = torch.tensor(pd.get_dummies(month).values)
W = torch.tensor(pd.get_dummies(win).values).float()
HAT = torch.transpose(torch.cat((torch.transpose(HT, 0, 1),torch.transpose(AT, 0, 1)),0), 0, 1).float()

HATM = torch.zeros(4180,72,10)
for n in range(4180):
    for i in range(72):
        for j in range(10):
            if M[n,j] or HAT[n,i] == 1:
                HATM[n,i,j] = 1
print(HATM)

total_step = 4180

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(0, HAT.shape[0]):
        # get the inputs
        inputs, labels = HATM[i], W[i]
        """mu, sigma = 0, 0.01
        noise = torch.from_numpy(np.random.normal(mu, sigma, [1,72])).float()
        inputs = inputs + noise"""
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.unsqueeze(0)
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            running_loss = 0.0
            print(outputs)
            print(labels)
print('Finished Training')
