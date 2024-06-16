#%% Importing the libraries

import pandas as pd
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#%% Loading the dataset

movies = pd.read_csv(
    'ml-1m/movies.dat', sep='::', \
        engine='python', encoding='latin-1', header=None)
    
users = pd.read_csv(
    'ml-1m/users.dat', sep='::', \
        engine='python', encoding='latin-1', header=None)

rating = pd.read_csv(
    'ml-1m/ratings.dat', sep='::', \
        engine='python', encoding='latin-1', header=None)
    
#%% Prepping the train and test set
train_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
train_set = np.array(train_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

#%% Finding the max numbers in the training and test set

nb_users = int(max(max(train_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(train_set[:,1]), max(test_set[:,1])))

#%% Data conversion function

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

train_set = convert(train_set)
test_set = convert(test_set)

#%% Converting the data to tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_set = torch.FloatTensor(train_set)
test_set =  torch.FloatTensor(test_set)
train_set = train_set.to(device)
test_set = test_set.to(device)


#%% Creating a Stacked Autoencoder

class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        # Encoding
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        
        # Decoding
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # Encoding
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        # Decoding
        x = self.activation(self.fc3(x))
        x =  self.fc4(x)
        
        return x

sae = SAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

#%% Training the SAE

nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_users):
    c_input = Variable(train_set[id_user]).unsqueeze(0).to(device)
    target = c_input.clone().to(device)
    if torch.sum(target.data > 0) > 0:
      output = sae(c_input)
      output.to(device)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += torch.sqrt(loss.data * mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))

#%% Testing the SAE

test_loss = 0
s = 0.
for id_user in range(nb_users):
  c_input = Variable(train_set[id_user]).unsqueeze(0).to(device)
  target = Variable(test_set[id_user]).unsqueeze(0).to(device)
  if torch.sum(target.data > 0) > 0:
    output = sae(c_input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += torch.sqrt(loss.data * mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))