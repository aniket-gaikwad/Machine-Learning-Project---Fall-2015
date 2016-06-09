import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
import RandomSampling as randomSampling
from sklearn  import linear_model
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

"""
	Name : Aniket Gaikwad
	Desc : Automated feature selection Methods.
"""

def RFE_featureSelection(X_train,Y_train):
	## Sampling
	RSObj=randomSampling.randomSampling()
	(X_train,Y_train)=RSObj.getRandomSample(X_train,Y_train)
	X_train.reset_index(drop=True,inplace=True)
	Y_train.reset_index(drop=True,inplace=True)

	## Select classifier and parameters
	logistic = linear_model.LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
	          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
	          penalty='l1', random_state=None, solver='liblinear', tol=0.01,
	          verbose=0, warm_start=False)

	## Initialiaze RFE
	rfecv = RFECV(estimator=logistic, step=1, cv=5,
	              scoring='recall')

	## Fit data
	rfecv.fit(X_train, Y_train)

	## Selected Features
	print("Optimal number of features : %d" % rfecv.n_features_)

	## Plot importance
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()

	#print('\n Selectd Columns : {0}').format(list(rfecv.support_))
	print('\n Selectd Columns : {0}').format(X_train.columns[list(rfecv.support_)])
	selected_columns = X_train.columns[list(rfecv.support_)]
	return selected_columns



def Boosting_featureSelection(X_train,Y_train):
	## Feature selection based on GradientBoostedTrees
	
	## Sampling
	RSObj=randomSampling.randomSampling()
	(X_train,Y_train)=RSObj.getRandomSample(X_train,Y_train)
	X_train.reset_index(drop=True,inplace=True)
	Y_train.reset_index(drop=True,inplace=True)

	## Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=100)

	## Fit Forest
	forest.fit(X_train, Y_train)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]

	## Print the feature ranking
	print("Feature ranking:")
	cols = list(X_train.columns)
	for f in range(X_train.shape[1]):
	    print("%d. feature %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))

	## Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X_train.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(X_train.shape[1]), [cols[i] for i in indices])
	plt.xlim([-1, X_train.shape[1]])
	plt.show()

	## Generate dictionary of column importance
	cols = [cols[i] for i in indices] 
	dictionary = dict(zip(cols,importances))
	return dictionary