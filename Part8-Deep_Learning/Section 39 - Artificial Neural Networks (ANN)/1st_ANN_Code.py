# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:37:01 2020

@author: Haravindan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
X = X[:,1:]
ct_1= ColumnTransformer([('encoder',OneHotEncoder(),[3])], remainder = 'passthrough')
X = np.array(ct_1.fit_transform(X))
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 6,kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(units = 6,kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

classifier.fit(X_train,y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
