#%% Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#%% Reading and splitting the dataset

df = pd.read_csv('Credit_Card_Applications.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Feature Scaling

sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#%% Training the SOM

from minisom import MiniSom
som = MiniSom(10, 10, 15)
som.random_weights_init(X)
som.train_random(X, 100)

#%% Plotting the SOM

plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()
markers, colors = ['o', 's'], ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, 
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor=colors[y[i]],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=2)
plt.show()

#%% Pulling the fraud data

mapping = som.win_map(X)
frauds = np.concatenate((mapping[(8,1)], mapping[(3,1)]), axis=0)
frauds = sc.inverse_transform(frauds)