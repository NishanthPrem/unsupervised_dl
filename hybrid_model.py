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
frauds = np.concatenate((mapping[(7,5)], mapping[(8,5)]), axis=0)
frauds = sc.inverse_transform(frauds)

#%% Creating the dependent variable

customers = df.iloc[:, 1:].values
is_fraud = np.zeros(len(df))

for i in range(len(df)):
    if df.iloc[i,0] in frauds:
        is_fraud[i] = 1

#%% Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

#%% Building the ANN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(
    units=6, kernel_initializer='uniform', activation='relu', input_dim=15))

classifier.add(Dense(
    units=6, kernel_initializer='uniform', activation='relu', input_dim=6))

classifier.add(Dense(
    units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(customers, is_fraud, batch_size=5, epochs=10)

#%% Predictions

y_pred = classifier.predict(customers)
y_pred = np.concatenate((df.iloc[:, 0:1].values, y_pred), axis=1)

#%% Sorting

y_pred = y_pred[y_pred[:,1].argsort()]