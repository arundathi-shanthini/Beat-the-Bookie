import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch



training_data=pd.read_csv('epl-training.csv')

win = training_data['FTR']

print(training_data)
print(win)
training = torch.tensor(training_data.values)
print(training)
