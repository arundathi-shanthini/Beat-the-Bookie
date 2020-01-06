#Importing all the libriaries
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

#defining the number of epochs that the code will run through
num_epochs = 50


#defining the architecture for the neural net using a self refferign class
class NeuralNetworkCalculator(nn.Module):
    def __init__(self):
        super(NeuralNetworkCalculator, self).__init__()
        self.layer_1 = torch.nn.Conv1d(1,1,kernel = 5,stride = 1)
        self.layer_2 = torch.nn.Conv1d(1,1,5,stride = 1)
        self.layer_3 = torch.nn.Conv1d(1,1,kernel = 5,padding = 2,stride = 1)
        self.layer_4 = torch.nn.Conv1d(1,1,kernel = 5,padding = 2,stride = 1)
        self.layer_5 = torch.nn.Conv1d(1,1,kernel = 5,padding = 2,stride = 1)
        self.layer_6 = torch.nn.Linear(11,3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        return x

#loading the data from the created feature csv file
training_data_features=pd.read_csv('feature_Export.csv')
#initialising the lablel encoder
le = preprocessing.LabelEncoder()
#itterating through the data frame to normalise all the values
for column in training_data_features:
    training_data_features[column] = training_data_features[column]/training_data_features[column].max()
#defining the output chriteria to leant on from the main csv provided originaly
training_data_winner=pd.read_csv('epl-training.csv')
win = training_data_winner['FTR']
#converting the winning team information to a tensor
W = torch.tensor(pd.get_dummies(win).values).float()
#covverign the training data to a tensor
FE = torch.tensor(training_data_features.values).float()
#initialising the model
model = NeuralNetworkCalculator()
#defining the loss chriterion as mean squred loss
criterion = nn.MSELoss()
#definign the optimiser for the backpropagation, Adam in this case
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
#setting the prossesing to the most efficient avalible device, CPU in my case
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#saving the model to the said device
model.to(device)
#generating the summary of the model as a printout
summary(model, input_size=(1, 19))

#splitting the data into training and test sets
training = 3971
test = 209
total_step = 4180

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(training):
        # get the inputs and labels relabled
        inputs, labels = FE[i], W[i]
        #making sure that the dimentions of the inputs and labels are the same as the models
        inputs = inputs.unsqueeze(0)
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        labels = labels.unsqueeze(0)
        #reset the grad function
        optimizer.zero_grad()
        #save the inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        #put intputs into the model and obtain outputs
        outputs = model(inputs)
        #obtain the loss between the outputs and actual labels
        loss = criterion(outputs, labels)
        #backpropogate and optimize
        loss.backward()
        optimizer.step()
        #keeping a tally of the total loss
        running_loss += loss.item()
        # print statistics for every 1000 inputs
        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
print('Finished Training')
print('Testing')
#test the model trained
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

    #outputs the persentage of correctly predicted games
    print('Test Accuracy of the model on the 209 games: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'CNN19.ckpt')
