import pandas as pd
import os
import algorithms as algs
import utilities as util
import RandomSampling as RS
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

"""
	Name : Aniket Gaikwad
	Desc : Utility functions for Running.
"""

class run:

	def __init__(self):
		"""
			Two types of features :
				1. HandPicked features
				2. Gradient Score based features

			Three methods :
				1. Without Sampling
				2. With Sampling
				3. With Weighing of class labels
			"""

	def loadcsv(self,fileName,fileType=None):
		dataset = pd.read_csv(fileName,sep=',',header=0)
		return dataset

	def loadData(self,fileName,fileType=None):
		dataset=None
		print('\n Loading data .....')
		dataset=self.loadcsv(fileName,fileType)
		print('\n Size of data : {0}').format(dataset.shape)
		return dataset

	def initialPreprocessing(self,trainDataset,testDataset):
		trainDataset.drop('Unnamed: 0',axis=1,inplace=True)
		testDataset.drop('Unnamed: 0',axis=1,inplace=True)
		Y_train = trainDataset['IsBadBuy']
		X_train = trainDataset.drop('IsBadBuy', axis=1)
		X_test = testDataset
		trainDataset = None
		testDataset = None
		column_all_nonzeros = list(X_train.columns[(X_train!=0).any()])
		X_train = X_train[column_all_nonzeros]
		X_test = X_test[column_all_nonzeros]
		return ((X_train,Y_train),(X_test))

	def randomShuffleData(self,X,Y):
		X_new = pd.concat([X,Y],axis=1)
		X_new = X_new.iloc[np.random.permutation(len(X_new))]
		Y = X_new['IsBadBuy']
		X = X_new.drop('IsBadBuy', axis=1)
		return (X,Y)

	def splitDataset(self,dataset,labels,trainsize=66, testsize=34, testfile=None):
		"""
        	Split dataset into size as per trainsize and testsize.
        	Return Train and Test dataset.
        """
	    
		print('\n Dataset size : {0}').format(dataset.shape[0])
		trainsize = dataset.shape[0] * trainsize / 100
		testsize = dataset.shape[0] * testsize / 100
		print('\n trainsize : {0}, TestSize : {1}').format(trainsize,testsize)
		randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
		X_train = dataset.ix[randindices[0:trainsize]]
		Y_train = labels.ix[randindices[0:trainsize]]
		X_test = dataset.ix[randindices[trainsize:trainsize+testsize]]
		Y_test = labels.ix[randindices[trainsize:trainsize+testsize]]
		return ((X_train,Y_train), (X_test,Y_test))

	def runClassifier(self,X_train,Y_train,X_test,Y_test):
		classalgs = {
		               'Logistic_Regression' : algs.LogisticRegression1(),
		               'Gauassian_SVM' :algs.SVM1(),
		               'Logistic_Regression_weighted ' : algs.LogisticRegression1wt(),
		               'Gauassian_SVM_weighted' :algs.SVM1wt()
		            }

		for learnername, learner in classalgs.iteritems():
		    print 'Running learner = ' + learnername
		    learner.learn(X_train, Y_train)
		    predictions = learner.predict(X_test)
		    recall = util.getRecall(Y_test, predictions)
		    print '\n Recall for ' + learnername + ': ' + str(recall)
		    precision=util.getPrecision(Y_test, predictions)
		    print '\n Precision for ' + learnername + ': ' + str(precision)
		    f5_score=util.getF5(precision,recall)
		    print '\n F5 Score for ' + learnername + ': ' + str(f5_score)
		    AUCROC =util.getAUCROC(Y_test, predictions)
		    print '\n AUC ROC Score for ' + learnername + ': ' + str(AUCROC)
		    AUCROCPlotPoints =util.getAUCROCPlotPoints(Y_test, predictions)
		    print('\n tpr : {0} fpr : {1} auc_roc : {2} learnername : {3}').format(AUCROCPlotPoints[0],AUCROCPlotPoints[1],AUCROCPlotPoints[2],learnername)
		    self.plotGraph(AUCROCPlotPoints,learnername)
		    f1_score = util.fscore(Y_test, predictions)
		    print '\n f1_score for ' + learnername + ': ' + str(f1_score)
		    accuracy = util.getaccuracy(Y_test, predictions)
		    print 'Accuracy for ' + learnername + ': ' + str(accuracy)


	def plotGraph(self,AUCROCPlotPoints,learnerName):
		"""
			Plot the AUC-ROC plot for each classifier.
		"""
		fpr=AUCROCPlotPoints[0]
		tpr=AUCROCPlotPoints[1]
		roc_auc= metrics.auc(fpr, tpr)

		plt.figure()
		plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example for : '+learnerName)
		plt.legend(loc="lower right")
		plt.show()




