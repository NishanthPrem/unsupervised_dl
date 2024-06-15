#%% Importing the libraries

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
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

nb_users = max(max(train_set[:,0]), max(test_set[:,0]))
nb_movies = max(max(train_set[:,1]), max(test_set[:,1]))

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

train_set = torch.FloatTensor(train_set)
test_set =  torch.FloatTensor(test_set)
        
#%% Converting the ratings to 0's and 1's

train_set[train_set == 0 ] = -1
train_set[train_set == 1 ] = 0
train_set[train_set == 2 ] = 0
train_set[train_set >= 3 ] = 1

test_set[test_set == 0 ] = -1
test_set[test_set == 1 ] = 0
test_set[test_set == 2 ] = 0
test_set[test_set >= 3 ] = 1

#%% Creating the Neural Network

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nv, nh)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        