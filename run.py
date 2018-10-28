#!/usr/bin/env python3
# Homework 1: Classification using Naive Bayesian & Voted Perceptron
# Created on 10/24/2018
# Author: Saolei
#-----------------------------------------------------------------------------------------------------------------------

# import the required packages here

import numpy as np
import csv, sys
import numpy.matlib as matlib
DEBUG=False
T    =50
# define other functions here

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
	'''The function to run your ML algorithm on given datasets, generate the predictions and save them into the provided file path
	
	Parameters
	----------
	Xtrain_file: string
		the path to Xtrain csv file
	Ytrain_file: string
		the path to Ytrain csv file
	test_data_file: string
		the path to test data csv file
	pred_file: string
		the prediction file to be saved by your code. You have to save your predictions into this file path following the same format of Ytrain_file
	'''
	## your implementation here
	# read data from Xtrain_file, Ytrain_file and test_data_file

	# your algorithm

	# save your predictions into the file pred_file
	print("I LOVE NLP!")
	Xtrain		=	np.loadtxt(Xtrain_file,delimiter=',',dtype=int)
	Ytrain		=	np.loadtxt(Ytrain_file,delimiter=',',dtype=int)
	Xtest		=	np.loadtxt(test_data_file,delimiter=',',dtype=int)
	Vac		= 	len(Xtrain[0])
	#if SHUFFLE==True:
	#	Shuffle_index	=	np.random.permutation(range(data_length)) 
	#else:
	#	Shuffle_index	=	range(data_length) 
	
	#if DEBUG==True:
	#	print("Shuffle:{}".format(Shuffle_index))
	#Xtrain		=	Xtrain[Shuffle_index,...]
	#Ytrain		=	Ytrain[Shuffle_index]
	#test_pct	=	0.9
	#train_pct	=	1.0
	#Xtrain		=	Xarray[0:int(train_pct*test_pct*data_length),...]
	#Xtest		=	Xarray[int(test_pct*data_length):,...]
	#Ytrain		=	Yarray[0:int(test_pct*data_length),...]
	#Ytest		=	Yarray[int(test_pct*data_length):,...]
	#Xtrain		=	read_traindata(Xtrain_file)
	#Ytrain		=	read_traindata(Ytrain_file)
	#DEBUGGING ONLY
	if(DEBUG==True):
		for i in range(0,1):
			print('i{}: Xtrian{}: Shape:'.format(i,Xtrain))
			print(Xtrain.shape)
			print(Xtrain.ndim)
			print(Xtrain.size)
			print(Xtrain[i].shape)
			print(Xtrain[i].ndim)
			print(type(Xtrain[i]))
			print(type(Xtrain[i,0]))
			print(type(Xtrain[i]))
			print(type(Xtrain[i][0]))
			print('i{}: Yarray{}: Shape:'.format(i,Ytrain))
			print(Ytrain.shape)
			print(Ytrain.ndim)
			print(Ytrain.size)
			print(Ytrain[i].shape)
			print(Ytrain[i].ndim)
			print(type(Ytrain[i]))
			print(type(Ytrain[i]))
			#print('i{}: Xtrain{}: Row{}: Col{}'.format(i,Xtrain[i],len(Xtrain),len(Xtrain[i])))
			#write_data(Xread_file,Xtrain)
			#print('i{}: Ytrain{}: Row{}: Col{}'.format(i,Ytrain[i],len(Ytrain),len(Ytrain[i])))
			#write_data(Yread_file,Ytrain)


	print("I LOVE LOVE NLP!!")
	##----------------------------------------------
	##--------Train Data Processing------------------
	##----------------------------------------------
	Y_count  = len(Ytrain)
	#Vac	 = len(X_1array[0])
	#W_0all   = [sum(x) for x in zip(*X_0array)]
	if(DEBUG==True):
		print('Ytrian type:{}'.format(type(Ytrain)))
		print('Xtrian type:{}'.format(type(Xtrain)))
		print('Vocabulary :'+str(Vac))
#---------------------------------------------------------------------
#-------------------------Trainning ----------------------------------
	c_all=[]
	c_all.append(0)	
	w_all=np.zeros(Vac)
	for t in range(T):
		for i,val in enumerate(Xtrain):
			if((Ytrain[i]-0.5)*sum(w_all[-1]*val) <= 0):
				w_all=np.vstack([w_all,w_all[-1]+2*(Ytrain[i]-0.5)*val])
				c_all.append(1)
			else:
				c_all[-1]+=1
	c_all=np.asarray(c_all)
	if DEBUG==True:
		print("w_all :{}".format(w_all))
		print("w_all shape:{}".format(w_all.shape))
		print("c_all :{}".format(c_all))
		print("c_all shape:{}".format(c_all.shape))

#---------------------------------------------------------------------
#------------------Prediction & Save----------------------------------
	#return []
	pred_y	=	[]
	for x in Xtest:
			xdotw = np.sum(x*w_all,axis=1)
			if DEBUG==True:
				print("x:{}".format(x.shape))	
				print("xdot_wall:{}".format(xdotw))	
				print("xdot_wall shape:{}".format(xdotw.shape))
			y=np.sign(np.sum(c_all*np.sign(xdotw)))	
			if y >=0:
				pred_y.append(int(1))
			else:
				pred_y.append(int(0))
	pred_y	=	np.asarray(pred_y)
	print("pred_y:{}".format(pred_y))
	print("Ytrain:{}".format(Ytrain))
	print("Xtrain:{}".format(Xtrain))
	np.savetxt(pred_file,pred_y,fmt='%d',delimiter=',')
	return pred_y

def GetScore(Ytest,pred_y):
	print("Give me a high score!!Pls!!!")
	# for binary classification only
	acc	=	1.0-float(np.sum(np.abs(Ytest-pred_y))/len(Ytest))
	d	=	np.sum(Ytest*pred_y)
	b_d	=	np.sum(pred_y)
	c_d	=	np.sum(Ytest)
	precision	=	float(d/b_d)
	recall		=	float(d/c_d)
	Fmea		=	2*precision*recall/(precision+recall)
	Final_score	=	0.5*acc+0.5*Fmea
	return acc,Fmea,Final_score
	


