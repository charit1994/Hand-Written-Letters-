#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 21:28:36 2017

@author: charith_gunuganti
"""

# Linear Regression (Linear Reg on ATNTFaceImages)
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
labels= data[:,0]
images= data[:,1:]
labels=labels.reshape(400,1)

from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test =train_test_split(images,labels,test_size=0.20)

#Just Replace X_Train=X_Train.T(for all .T's) , kept it like this so that you can understand 
X_TrainTrans = X_Train.T
X_TestTrans=X_Test.T
Y_TrainTrans=Y_Train.T
Y_TestTrans =Y_Test.T



s=(40,320)
y_indicator_matrix = np.zeros(s)

for i in range(0,36):
        y_indicator_matrix[Y_Train[i]-1,i] = 1;
A_train = np.ones((1,320))    # N_train : number of training instance
A_test = np.ones((1,80))      # N_test  : number of test instance
Xtrain_padding = np.row_stack((X_TrainTrans,A_train))
Xtest_padding = np.row_stack((X_TestTrans,A_test))

Y=[1,4,5,2,3]

'''computing the regression coefficients'''
B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), y_indicator_matrix.T)   # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix
Ytest_padding = np.dot(B_padding.T,Xtest_padding)
Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
Ytest_padding_argmax = Ytest_padding_argmax.reshape(80,1)
'''Comparsion you try ra''' #I FeelSleepy
err_test_padding = Y_Test - Ytest_padding_argmax
TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100


