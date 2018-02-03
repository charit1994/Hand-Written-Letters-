#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:46:38 2017

@author: charith_gunuganti
"""

# SVM ( on ATNTFaceImages)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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
from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X, y)


#Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5)
print "The 5 fold CV accuracy scores are:",scores
print "\n The mean classifier accuracy is", scores.mean()
