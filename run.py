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
# define other functions here
def read_traindata(file_name):
	data	=	[]
	temp	=	[]
	with open(file_name) as csvfile:
		csvreader = csv.reader(csvfile)
		try:
			for row in csvreader:
				#for rowi in row:
				#	print(rowi)
				#	temp.append(int(rowi))
				if (len(row)>1):
					data.append([int(temp) for temp in row ])
				else:
					data.append(int(row[0]))
				#print(', '.join(row))

		except csv.Error as e:
			sys.exit('file {},line {}:{}'.format(Xtrain_file,csvreader.line_num,e))
	return data

def write_data(file_name,someiterable):
	with open(file_name, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(someiterable)


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
	#DEBUGGING ONLY
	if(DEBUG==True):
		for i in range(0,1):
			print('i{}: Xtrain{}: Shape:'.format(i,Xtrain))
			print(Xtrain.shape)
			print(Xtrain.ndim)
			print(Xtrain.size)
			print(Xtrain[i].shape)
			print(Xtrain[i].ndim)
			print(type(Xtrain[i]))
			print(type(Xtrain[i,0]))
			print(type(Xtrain[i]))
			print(type(Xtrain[i][0]))
			print('i{}: Ytrian{}: Shape:'.format(i,Ytrain))
			print(Ytrain.shape)
			print(Ytrain.ndim)
			print(Ytrain.size)
			print(Ytrain[i].shape)
			print(Ytrain[i].ndim)
			print(type(Ytrain[i]))
			print(type(Ytrain[i]))

	print("I LOVE LOVE NLP!!")
	##----------------------------------------------
	##--------Train Data Processing------------------
	##----------------------------------------------
	Y_1index = [i for (i,val) in enumerate(Ytrain) if val>0]
	Y_0index = [i for (i,val) in enumerate(Ytrain) if val<1]
	X_1array = Xtrain[Y_1index,:]
	X_0array = Xtrain[Y_0index,:]
	Y_count  = len(Ytrain)
	Y_1count = len(Y_1index)
	Y_0count = len(Y_0index)
	W_1all   = np.asarray([sum(x) for x in zip(*X_1array)])
	W_0all   = np.asarray([sum(x) for x in zip(*X_0array)])
	W_all    = W_1all+W_0all
	if(DEBUG==True):
		print('Ytrain type:{}'.format(type(Ytrain)))
		print('Y_1index:{}'.format(Y_1index))
		print('X_1array:{}'.format(X_1array))
		print('X_1array shape:{}'.format(X_1array.shape))
		print('X_0array:{}'.format(X_0array))
		print('X_0array shape:{}'.format(X_0array.shape))
		print('Y_1count:{}'.format(Y_1count))
		print('Y_0count:{}'.format(Y_0count))
		print('Y_count:{}'.format(Y_count))
		print('W_1all:{}'.format(W_1all))
		print('W_0all:{}'.format(W_0all))
		print('W_all:{}'.format(W_all))
		print('W_1all shape:'+str(np.shape(W_1all)))
		print('W_0all shape:'+str(np.shape(W_0all)))
		print('W_all shape:'+str(np.shape(W_all)))
		print('Vocabulary :'+str(Vac))
	W_notzero =	np.asarray([i for (i,val) in enumerate(W_all) if val>0])
	#C1_prob	=	(1.0+W_1all)/(np.sum(W_1all)+Vac)
	#C0_prob	=	(1.0+W_0all)/(np.sum(W_0all)+Vac)
	C1toC0	=	((float(1.0)+W_1all)/(float(1.0)+W_0all))*((np.sum(W_0all)+float(Vac))/(np.sum(W_1all)+float(Vac)))
	Test_index =	[]
	for x in np.nditer(Xtest):
		if x >0:
			Test_index.append(1.0)
		else:
			Test_index.append(0.0)
	Test_index=np.asarray(Test_index)
	Test_index=Test_index.reshape(Xtest.shape)
	if DEBUG==True:
		print("test_index:{}".format(Test_index))
		print("test_index:{}".format(Test_index.shape))
	#P_y1	=	np.sum(np.log10(C1_prob)*Xtest,axis=1)+np.log10(float(Y_1count/Y_count))
	#P_y0	=	np.sum(np.log10(C0_prob)*Xtest,axis=1)+np.log10(float(Y_0count/Y_count))
	#P_diff	=	P_y1-P_y0
	#P_diff	=	np.sum(np.log10(C1toC0)*Test_index,axis=1)+np.log10(float(Y_1count)/float(Y_0count))
	#P_diff	=	np.sum(np.log10(C1toC0)*(Xtest+1),axis=1)/(np.sum(Xtest,axis=1)+Vac)+np.log10(float(Y_1count)/float(Y_0count))
	#P_diff	=	np.sum(np.log10(C1toC0)/(1.0+W_all)*Y_count*Xtest,axis=1)+np.log10(float(Y_1count)/float(Y_0count))
	#P_diff	=	np.sum(np.log10(C1toC0)*Test_index,axis=1)+np.log10(float(Y_1count)/float(Y_0count))
	#P_diff	=	np.sum(np.log10(C1toC0)*Xtest,axis=1)+np.log10(float(Y_1count)/float(Y_0count))
	P_diff	=	np.matmul(Xtest,np.log10(C1toC0))+np.log10(float(Y_1count)/float(Y_0count))
	pred_y	=	[]
	for x in np.nditer(P_diff):
		if x >=0:
			pred_y.append(int(1))
		else:
			pred_y.append(int(0))
	pred_y	=	np.asarray(pred_y)
	np.savetxt(pred_file,pred_y,fmt='%d',delimiter=',')
	if DEBUG==True:
		print("P_y1-y0:{}".format(P_diff))
		print("P_y1-y0:shape{}".format(P_diff.shape))
		print("Ypred:{}".format(pred_y))
		print("W_1all:{}".format(W_1all.sum()))
		print("W_1all:{}".format(np.sum(W_1all)))
		print("W_0all:{}".format(W_0all.sum()))
		print("W_notzero:{}".format(W_notzero))
		print("W_notzero shape:{}".format(W_notzero.shape))
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
	


