#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:41:23 2017

@author: charith_gunuganti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
df = pd.read_csv('trainDataXY.txt')
dfw = pd.read_csv('testDataX.txt', header = None)
dfw=dfw.transpose()
X_test = dfw.iloc[:,:].values


#Values for matrix of features X,y
df.columns.values[0:9] = '1'
df.columns.values[9:18] = '2'
df.columns.values[18:27] = '3'
df.columns.values[27:36] = '4'
df.columns.values[36:45] = '5'


df = df.transpose()
X= df.iloc[:, :].values
df.reset_index(inplace=True)
y = df.iloc[:, 0].values
y=y.astype(int)



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.fit_transform(X_test)


#fitting the classifier to the training set

'''#LINEAR REGRSSION
#LBFGS Is a linear solver
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,solver='lbfgs',multi_class='multinomial')
classifier.fit(X, y)
y_pred=classifier.predict(X_test)


#CENTROID
from sklearn.neighbors import NearestCentroid
classifier = NearestCentroid()
classifier.fit(X, y)
y_pred=classifier.predict(X_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X, y)
y_pred=classifier.predict(X_test)

# SVM 
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X, y)
y_pred=classifier.predict(X_test)'''

#CLASSIFER Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5)
print "The 5 fold CV accuracy scores are:",scores
print "\n The mean classifier accuracy is", scores.mean()
