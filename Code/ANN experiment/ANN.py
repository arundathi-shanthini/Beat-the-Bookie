#import libriaries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

#number of trining epochs
num_epochs = 50

#defining the architecture for the neural net using a self refferign class
class NeuralNetworkCalculator(nn.Module):
    def __init__(self):
        super(NeuralNetworkCalculator, self).__init__()
        self.layer_1 = torch.nn.Linear(72, 36)
        self.layer_2 = torch.nn.Linear(36, 36)
        self.layer_3 = torch.nn.Linear(36, 18)
        self.layer_4 = torch.nn.Linear(18, 18)
        self.layer_5 = torch.nn.Linear(18, 9)
        self.layer_6 = torch.nn.Linear(9, 3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        return x
#Initialising the model
net = NeuralNetworkCalculator()
#defining the loss criterion mean squred loss in this case
criterion = nn.MSELoss()
#defining the optimiser for back propagation Adam in this case
optimizer = optim.Adam(net.parameters(), lr = 0.00001)
#choosing the most optimal device to run the training on, CPU in this case
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#save the archetecture to the selected device
net.to(device)
#produce summary for the archetecture
summary(net, input_size=(1, 72))
#load the training data
training_data=pd.read_csv('epl-training.csv')
#separate the home temas away teams and the winning information
win = training_data['FTR']
hometeam = training_data['HomeTeam']
awayteam = training_data['AwayTeam']
#convert the data to tensors that can be used in training
HT = torch.tensor(pd.get_dummies(hometeam).values)
AT = torch.tensor(pd.get_dummies(awayteam).values)
W = torch.tensor(pd.get_dummies(win).values).float()
HAT = torch.transpose(torch.cat((torch.transpose(HT, 0, 1),torch.transpose(AT, 0, 1)),0), 0, 1).float()

#splitting the data into training and test sets
training = 3971
test = 209
total_step = 4180

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(0, training):
        # get the inputs
        inputs, labels = HAT[i], W[i]
        #experimented with adding random gausian noise
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

        # print statistics for every 1000 training samples
        running_loss += loss.item()
        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            running_loss = 0.0
print('Finished Training')

#due to poor succsess, aparent from the stats printout, i didnt test the outcome not did I save the model 
