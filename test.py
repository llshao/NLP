#!/usr/bin/env python3
# Homework 1: Classification using Naive Bayesian & Voted Perceptron
# Created on 10/24/2018
# Author: Saolei
#-----------------------------------------------------------------------------------------------------------------------

# import the required packages here

import numpy as np
import csv, sys
import numpy.matlib as matlib
from run import run
from run import GetScore

# Gloable variable & setting

DEBUG=False
SHUFFLE=False
np.random.seed(0)

#===============================================================================
#===============================================================================
#-----------------------------Main Function-------------------------------------
#-------------------------------------------------------------------------------
X_file	=	'../reference/Xtrain.csv'
Y_file	=	'../reference/Ytrain.csv'
test_pct	=	0.9		#from test_pct to end will be used as the test set
train_pct	=	0.6		#how much percentage of 0-test_pct will be used as the training set
Xtrain_file	=	'Xtrain'+str(train_pct)+'.csv'
Ytrain_file	=	'Ytrain'+str(train_pct)+'.csv'
test_data_file  =	'test.csv'
pred_file	=	'Ypred'+str(train_pct)+'.csv'

#Xtrain		=	[]
#Ytrain		=	[]
#Yarray 		=	np.asarray(read_traindata(Ytrain_file))
#Xarray 		=	np.asarray(read_traindata(Xtrain_file))
Xarray		=	np.loadtxt(X_file,delimiter=',',dtype=int)
Yarray		=	np.loadtxt(Y_file,delimiter=',',dtype=int)
data_length	=	len(Xarray)
Vac		= 	len(Xarray[0])
if SHUFFLE==True:
	Shuffle_index	=	np.random.permutation(range(data_length)) 
else:
	Shuffle_index	=	range(data_length) 

if DEBUG==True:
	print("Shuffle:{}".format(Shuffle_index))
Xarray		=	Xarray[Shuffle_index,...]
Yarray		=	Yarray[Shuffle_index]
Xtrain		=	Xarray[0:int(train_pct*test_pct*data_length),...]
Xtest		=	Xarray[int(test_pct*data_length):,...]
Ytrain		=	Yarray[0:int(train_pct*test_pct*data_length),...]
Ytest		=	Yarray[int(test_pct*data_length):,...]
#--------------Save as csv--------------------------------------------------
np.savetxt(Xtrain_file,Xtrain,fmt='%d',delimiter=',')
np.savetxt(Ytrain_file,Ytrain,fmt='%d',delimiter=',')
np.savetxt(test_data_file,Xtest,fmt='%d',delimiter=',')
pred_y=run(Xtrain_file, Ytrain_file, test_data_file, pred_file)
acc,Fmea,Final_score=GetScore(Ytest,pred_y)
print("acc"+str(acc)+" Fmea"+str(Fmea)+" Final_score"+str(Final_score))
