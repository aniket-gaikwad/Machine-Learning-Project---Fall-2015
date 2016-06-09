import matplotlib.pyplot as plt
import pandas as pd
import os
import utilities as util
import RandomSampling as RS
import numpy as np
import run_util as run_util
import Automated_FeatureSelection as AF
from sklearn.cross_validation import KFold
"""
	Name : Aniket Gaikwad
	Desc : This code will run different experimental runs.
"""
def loadData(runObj):
	
	dirPath = 'D:\\Fall 2015\\Machine Learning\\Project\\B555-Project\\MODIFICATIONS'
	
	## Load data and split into X and Y
	trainDataset = runObj.loadData(os.path.join(dirPath,'HandPickedFeaturesTrain1.csv'),'train')
	testDataset = runObj.loadData(os.path.join(dirPath,'HandPickedFeaturesTest.csv'),'test')
	return runObj.initialPreprocessing(trainDataset,testDataset)

def CrossValidationWithSampling():
	runObj = run_util.run()

	## Load data
	((X_train,Y_train),(X_test)) = loadData(runObj)
	
	## Sample train data 
	RSObj=RS.randomSampling()
	(X_train,Y_train) = RSObj.getRandomSample(X_train,Y_train)
	(X_train,Y_train) = runObj.randomShuffleData(X_train,Y_train)
	print('\n Size of X_train : {0}').format(X_train.shape)
	print('\n Size of Y_train : {0}').format(Y_train.shape)
	X_train.reset_index(drop=True,inplace=True)
	Y_train.reset_index(drop=True,inplace=True)

	## Cross Validation
	kf = KFold(X_train.shape[0], n_folds=5)
	for train_index, test_index in kf:
		X_train_CV, X_test_CV = X_train.ix[train_index], X_train.ix[test_index]
		Y_train_CV, Y_test_CV = Y_train.ix[train_index], Y_train.ix[test_index]
		print('\n Size of X_train_CV : {0}').format(X_train_CV.shape)
		print('\n Size of Y_train_CV : {0}').format(Y_train_CV.shape)
		print('\n Size of X_test_CV : {0}').format(X_test_CV.shape)
		print('\n Size of Y_test_CV : {0}').format(Y_test_CV.shape)

		## Run classifiers
		runObj.runClassifier(X_train_CV,Y_train_CV,X_test_CV,Y_test_CV)


def RFEWithSampling():
	runObj = run_util.run()

	## Load data
	((X_train,Y_train),(X_test)) = loadData(runObj)
	
	## Sample train data 
	RSObj=RS.randomSampling()
	(X_train,Y_train)=RSObj.getRandomSample(X_train,Y_train)
	(X_train,Y_train) = runObj.randomShuffleData(X_train,Y_train)
	print('\n Size of Sample X_train : {0}').format(X_train.shape)
	print('\n Size of Sample Y_train : {0}').format(Y_train.shape)
	X_train.reset_index(drop=True,inplace=True)
	Y_train.reset_index(drop=True,inplace=True)

	## RFE Feature Selection
	print('\n RFE Feature Selection starts...')
	selected_columns = AF.RFE_featureSelection(X_train,Y_train)
	print('\n RFE Feature Selection ends...')
	X_train = X_train[selected_columns]
	X_test = X_train[selected_columns]

	## Split data into 66% train and 33% test
	((X_train_S,Y_train_S), (X_test_S,Y_test_S)) = runObj.splitDataset(X_train,Y_train)
	print('\n Size of X_train_S : {0}').format(X_train_S.shape)
	print('\n Size of Y_train_S : {0}').format(Y_train_S.shape)
	print('\n Size of X_test_S : {0}').format(X_test_S.shape)
	print('\n Size of Y_test_S : {0}').format(Y_test_S.shape)

	## Run classifiers
	runObj.runClassifier(X_train_S,Y_train_S,X_test_S,Y_test_S)


def BoostingWithSampling():
	runObj = run_util.run()

	## Load data
	((X_train,Y_train),(X_test)) = loadData(runObj)
	
	## Sample train data 
	RSObj=RS.randomSampling()
	(X_train,Y_train)=RSObj.getRandomSample(X_train,Y_train)
	(X_train,Y_train) = runObj.randomShuffleData(X_train,Y_train)
	print('\n Size of X_train : {0}').format(X_train.shape)
	print('\n Size of Y_train : {0}').format(Y_train.shape)
	X_train.reset_index(drop=True,inplace=True)
	Y_train.reset_index(drop=True,inplace=True)

	## Boosting Feature Selection
	print('\n Boosting Feature Selection starts...')
	selected_columns_dict = AF.Boosting_featureSelection(X_train,Y_train)
	print('\n Boosting Feature Selection ends...')
	selected_columns=[]
	for col, imp in selected_columns_dict.iteritems():
		if(imp > 0.005):
			selected_columns.append(col)
	print('\n Selected Cols : {0}').format(len(selected_columns))
	
	X_train = X_train[selected_columns]
	X_test = X_train[selected_columns]

	## Split data into 66% train and 33% test
	((X_train_S,Y_train_S), (X_test_S,Y_test_S)) = runObj.splitDataset(X_train,Y_train)
	print('\n Size of X_train_S : {0}').format(X_train_S.shape)
	print('\n Size of Y_train_S : {0}').format(Y_train_S.shape)
	print('\n Size of X_test_S : {0}').format(X_test_S.shape)
	print('\n Size of Y_test_S : {0}').format(Y_test_S.shape)

	## Run classifiers
	runObj.runClassifier(X_train_S,Y_train_S,X_test_S,Y_test_S)

def handpickWithoutSampling():
	runObj = run_util.run()

	## Load data
	((X_train,Y_train),(X_test)) = loadData(runObj)

	## Split data into 66% train and 33% test
	((X_train_S,Y_train_S), (X_test_S,Y_test_S)) = runObj.splitDataset(X_train,Y_train)
	print('\n Size of X_train_S : {0}').format(X_train_S.shape)
	print('\n Size of Y_train_S : {0}').format(Y_train_S.shape)
	print('\n Size of X_test_S : {0}').format(X_test_S.shape)
	print('\n Size of Y_test_S : {0}').format(Y_test_S.shape)

	## Run classifiers
	runObj.runClassifier(X_train_S,Y_train_S,X_test_S,Y_test_S)            

def costValuesBasedPrediction():
	runObj = run_util.run()

	## Load data
	((X_train,Y_train),(X_test)) = loadData(runObj)

	## Select feature related to cost
	costFeatures = ['MMRAcquisitionAuctionAveragePrice','VehBCost']
	X_train = X_train[costFeatures]
	print('\n Size of X_train only cost features : {0}').format(X_train.shape)

	## Sample train data 
	RSObj=RS.randomSampling()
	(X_train,Y_train) = RSObj.getRandomSample(X_train,Y_train)
	(X_train,Y_train) = runObj.randomShuffleData(X_train,Y_train)
	print('\n Size of X_train : {0}').format(X_train.shape)
	print('\n Size of Y_train : {0}').format(Y_train.shape)
	X_train.reset_index(drop=True,inplace=True)
	Y_train.reset_index(drop=True,inplace=True)

	## Split data into 66% train and 33% test
	((X_train_S,Y_train_S), (X_test_S,Y_test_S)) = runObj.splitDataset(X_train,Y_train)
	print('\n Size of X_train_S : {0}').format(X_train_S.shape)
	print('\n Size of Y_train_S : {0}').format(Y_train_S.shape)
	print('\n Size of X_test_S : {0}').format(X_test_S.shape)
	print('\n Size of Y_test_S : {0}').format(Y_test_S.shape)

	## Run classifiers
	runObj.runClassifier(X_train_S,Y_train_S,X_test_S,Y_test_S)

def handpickWithSampling():
	runObj = run_util.run()

	## Load data
	((X_train,Y_train),(X_test)) = loadData(runObj)
	
	## Sample train data 
	RSObj=RS.randomSampling()
	(X_train,Y_train) = RSObj.getRandomSample(X_train,Y_train)
	(X_train,Y_train) = runObj.randomShuffleData(X_train,Y_train)
	print('\n Size of X_train : {0}').format(X_train.shape)
	print('\n Size of Y_train : {0}').format(Y_train.shape)
	X_train.reset_index(drop=True,inplace=True)
	Y_train.reset_index(drop=True,inplace=True)

	## Split data into 66% train and 33% test
	((X_train_S,Y_train_S), (X_test_S,Y_test_S)) = runObj.splitDataset(X_train,Y_train)
	print('\n Size of X_train_S : {0}').format(X_train_S.shape)
	print('\n Size of Y_train_S : {0}').format(Y_train_S.shape)
	print('\n Size of X_test_S : {0}').format(X_test_S.shape)
	print('\n Size of Y_test_S : {0}').format(Y_test_S.shape)

	## Run classifiers
	runObj.runClassifier(X_train_S,Y_train_S,X_test_S,Y_test_S)

if __name__ == "__main__":

	# 1. HandpickedFeatures + Sampling
	handpickWithSampling()

	# 2. HandpickedFeatures + WithoutSampling
	handpickWithoutSampling()

	# 3. RFE Feature Selection + Sampling
	RFEWithSampling()

	# 4. Boosting Tree based feature Selection +  Sampling	
	BoostingWithSampling()

	# 5. N fold CrossFold Validation
	CrossValidationWithSampling()

	# 6. Experiement with all cost values
	costValuesBasedPrediction()


