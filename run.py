#!/usr/bin/env python

# import the required packages here

# Homework 1: Classification using Naive Bayesian & Voted Perceptron
# Created on 10/24/2018
# Author: Leilai Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import csv, sys
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import scipy.stats as scist
import scipy.io as sio

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
	print("Hello NLP!\n")
	## your implementation here
	# read data from Xtrain_file, Ytrain_file and test_data_file

	# your algorithm

	# save your predictions into the file pred_file


# define other functions here
def read_traindata(file_name):
	data	=	[]
	with open(file_name, newline='') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		try:
			for row in csvreader:
				data.append(row)
				#print(', '.join(row))

		except csv.Error as e:
			sys.exit('file {},line {}:{}'.format(Xtrain_file,csvreader.line_num,e))
	return data

def write_data(file_name,someiterable):
	with open(file_name, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(someiterable)

print("Hello NLP\n")
Xtrain_file	=	"Xtrain.csv"
Ytrain_file	=	"Ytrain.csv"
Xread_file	=	"Test_readX.csv"
Yread_file	=	"Test_readY.csv"
Xtrain		=	[]
Ytrain		=	[]
Xarray 		=	np.asarray(read_traindata(Xtrain_file))
Xtrain		=	read_traindata(Xtrain_file)
Ytrain		=	read_traindata(Ytrain_file)
for i in range(0,2):
	print('i{}: Xarray{}: Shape:'.format(i,Xarray[i]))
	print(Xarray.shape)
	print(Xarray.ndim)
	print(Xarray.size)
	print(Xarray[i].shape)
	print(Xarray[i].ndim)
	print(type(Xarray[i]))
	print(type(Xtrain[i]))
	print("\n")
	#print('i{}: Xtrain{}: Row{}: Col{} \n'.format(i,Xtrain[i],len(Xtrain),len(Xtrain[i])))
	#write_data(Xread_file,Xtrain[i])
	#print('i{}: Ytrain{}: Row{}: Col{} \n'.format(i,Ytrain[i],len(Ytrain),len(Ytrain[i])))
	#write_data(Yread_file,Ytrain[i])


