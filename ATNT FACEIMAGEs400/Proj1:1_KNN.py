#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:53:57 2017

@author: charith_gunuganti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#KNN
# Importing the dataset 
def readData( file):
        train_file = open(file,'r')
        trainMatrix =[]
        for line in train_file:
            lineMatrix = list(map(int, line.split(',')))
            trainMatrix.append(lineMatrix)
        return (np.array(trainMatrix))
    
data = readData('ATNTFaceImages400.txt')
data=data.T
y= data[:,0]
X= data[:,1:]


#fitting the classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X, y)

#Accuracy k fold
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5)
print "The 5 fold CV accuracy scores are:",scores
print "\n The mean classifier accuracy is", scores.mean()





