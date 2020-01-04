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

num_epochs = 20

class NeuralNetworkCalculator(nn.Module):
    def __init__(self):
        super(NeuralNetworkCalculator, self).__init__()
        self.layer_1 = torch.nn.Conv1d(1,1,5,stride = 1)
        self.layer_2 = torch.nn.Conv1d(1,1,5,stride = 1)
        self.layer_3 = torch.nn.Conv1d(1,1,5,padding = 2,stride = 1)
        self.layer_4 = torch.nn.Conv1d(1,1,5,padding = 2,stride = 1)
        self.layer_5 = torch.nn.Conv1d(1,1,5,padding = 2,stride = 1)
        self.layer_6 = torch.nn.Linear(11,3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        return x


training_data_features=pd.read_csv('feature_Export.csv')

le = preprocessing.LabelEncoder()

for column in training_data_features:
    training_data_features[column] = training_data_features[column]/training_data_features[column].max()

training_data_winner=pd.read_csv('epl-training.csv')
win = training_data_winner['FTR']
W = torch.tensor(pd.get_dummies(win).values).float()
FE = torch.tensor(training_data_features.values).float()

model = NeuralNetworkCalculator()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model, input_size=(1, 19))


training = 3971
test = 209
total_step = 4180

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(FE.shape[0]):
        # get the inputs
        inputs, labels = FE[i], W[i]
        inputs = inputs.unsqueeze(0)
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        labels = labels.unsqueeze(0)

        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)


        #print(outputs.size())
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


        # print statistics
        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
print('Finished Training')
print('Testing')

with torch.no_grad():
    correct = 0
    total = 0
    for i in range(training, total_step):
        inputs, labels = FE[i], W[i]
        inputs = inputs.unsqueeze(0)
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        value1, index1 = torch.max(outputs,0)
        value2, index2 = torch.max(labels,0)
        if index1 == index2:
            correct = correct + 1
        total = total + 1


    print('Test Accuracy of the model on the 209 games: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'CNN19.ckpt')
