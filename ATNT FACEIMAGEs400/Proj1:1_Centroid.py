#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:27:22 2017

@author: charith_gunuganti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)'''

#test_train_split

#Centroid


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
from sklearn.neighbors import NearestCentroid
classifier = NearestCentroid()
classifier.fit(X, y)

#Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5)
print "The 5 fold CV accuracy scores are:",scores
print "\n The mean classifier accuracy is", scores.mean()
