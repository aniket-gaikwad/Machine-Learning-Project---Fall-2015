from sklearn.naive_bayes import MultinomialNB
from sklearn  import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import BernoulliRBM
import numpy as np
import utilities as utils
import itertools
from sklearn.grid_search import GridSearchCV
import pickle
import pandas as pd

"""

    Name : Aniket Gaikwad
    Desc : This is the class file for different classifier used. Each classifier class
           has three definitions viz. Initialization, Learning and Testing.
    	   Initialization will initialize the hyperparameters.
    	   Learning will build a model.
    	   Testing will run the test data on model.

"""

class NaiveBayes:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        model=MultinomialNB(fit_prior=True)
        model.fit(Xtrain,Ytrain)
        self.model=model

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction


class LogisticRegression:
    def __init__(self,C):
        self.model=None
        self.prediction=None
        self.C=C

    def learn(self,Xtrain,Ytrain):
        model=linear_model.LogisticRegression(penalty='l2',C=self.C,tol=0.01,class_weight={1.0:3.0,0.0:1.0})
        model.fit(Xtrain,Ytrain)
        self.model=model

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class LogisticRegression1wt:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        param_grid  = {'penalty' : ['l2','l1'],
                        'C' : [0.001, 0.01, 0.1, 10, 20],
                        'tol' : [0.001, 0.01, 0.1],
                        'class_weight' : [{1.0:3.0,0.0:1.0},{1.0:10.0,0.0:1.0}]
                        }
        est = linear_model.LogisticRegression()
        gs_cv = GridSearchCV(est,param_grid).fit(Xtrain,Ytrain)
        print("\n Best Para : ")
        best_params=gs_cv.best_params_
        print(best_params)
        best_estimator=gs_cv.best_estimator_
        print(best_estimator)
        with open("best_estimator_Wt_logistic","wb") as f:
            pickle.dump(best_estimator,f)
        self.model=best_estimator

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class LogisticRegression1:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        param_grid  = {'penalty' : ['l2','l1'],
                        'C' : [0.001, 0.01, 0.1, 10, 20],
                        'tol' : [0.001, 0.01, 0.1]
                        #,'class_weight' : None
                        }
        est = linear_model.LogisticRegression()
        gs_cv = GridSearchCV(est,param_grid).fit(Xtrain,Ytrain)
        print("\n Best Para : ")
        best_params=gs_cv.best_params_
        print(best_params)
        best_estimator=gs_cv.best_estimator_
        print(best_estimator)
        with open("best_estimator_logistic","wb") as f:
            pickle.dump(best_estimator,f)
        self.model=best_estimator

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class GradientBoost:
    def __init__(self,train_column_headers,n_estimators=None,fold_index=-1):
        self.model=None
        self.prediction=None
        self.n_estimators=n_estimators
        self.train_column_headers=train_column_headers
        self.fold_index=fold_index

    def learn(self,Xtrain,Ytrain):
        param_grid  = {'learning_rate' : [0.01],
                        'max_depth' : [4],
                        'min_samples_leaf' : [3, 5],
                        'max_features' : [1.0],
                        'subsample' : [0.5]}
        est = GradientBoostingClassifier(n_estimators=self.n_estimators)
        print('\n Number of estimators : {0}').format(self.n_estimators)
        gs_cv = GridSearchCV(est,param_grid).fit(Xtrain,Ytrain)
        print("\n Best Para : ")
        best_params=gs_cv.best_params_
        print(best_params)
        with open("best_param.pkl","wb") as f:
            pickle.dump(best_params,f)
        print("\n best_estimator_ : ")
        best_estimator=gs_cv.best_estimator_
        print(best_estimator)
        with open("best_estimator.pkl","wb") as f:
            pickle.dump(best_estimator,f) 
        
        feature_importances=best_estimator.feature_importances_
        feat_imp = pd.Series(feature_importances, self.train_column_headers).sort_values(ascending=False)
        print("\n Important Features : ")
        print(feat_imp)
        with open("feature_importances.pkl","wb") as f:
            pickle.dump(feat_imp,f)
        self.model=best_estimator



class GradientBoost1:
    def __init__(self,train_column_headers,n_estimators=None,fold_index=-1):
        self.model=None
        self.prediction=None
        self.n_estimators=n_estimators
        self.train_column_headers=train_column_headers
        self.fold_index=fold_index

    def learn(self,Xtrain,Ytrain):
        param_grid  = {'learning_rate' : [0.01],
                        'max_depth' : [4],
                        'min_samples_leaf' : [9],
                        'max_features' : [0.3],
                        'subsample' : [0.5]}
        est = GradientBoostingClassifier(n_estimators=self.n_estimators)
        print('\n Number of estimators : {0}').format(self.n_estimators)
        gs_cv = GridSearchCV(est,param_grid).fit(Xtrain,Ytrain)
        print("\n Best Para : ")
        best_params=gs_cv.best_params_
        print(best_params)
        with open("best_param_"+str(self.fold_index),"wb") as f:
            pickle.dump(best_params,f)
        print("\n best_estimator_ : ")
        best_estimator=gs_cv.best_estimator_
        print(best_estimator)
        with open("best_estimator_"+str(self.fold_index),"wb") as f:
            pickle.dump(best_estimator,f) 
        
        #model=GradientBoostingClassifier(learning_rate =0.1,n_estimators=self.n_estimators)
        #best_estimator.fit(Xtrain,Ytrain)
        feature_importances=best_estimator.feature_importances_
        feat_imp = pd.Series(feature_importances, self.train_column_headers).sort_values(ascending=False)
        print("\n Important Features : ")
        print(feat_imp)
        with open("feature_importances_"+str(self.fold_index),"wb") as f:
            pickle.dump(feature_importances,f)
        self.model=best_estimator


    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class DeepBeliefNetwork:
    def __init__(self,hiddenLayers,learning_rate):
        self.model=None
        self.data=None
        self.hiddenLayers=hiddenLayers
        self.learning_rate=learning_rate

    def learn(self,Xtrain,Ytrain):
        model = BernoulliRBM(n_components=self.hiddenLayers,learning_rate=self.learning_rate)
        self.model=model.fit(Xtrain,Ytrain)
        return self.model

    def transform(self,data):
        model=self.model
        self.data=model.transform(data)
        return self.data



class SVM1:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        param_grid  = {'kernel' : ['rbf'],
                        'C' : [0.001, 0.1]
                        #,'class_weight' : None
                        }
        est = svm.SVC()
        gs_cv = GridSearchCV(est,param_grid).fit(Xtrain,Ytrain)
        print("\n Best Para : ")
        best_params=gs_cv.best_params_
        print(best_params)
        best_estimator=gs_cv.best_estimator_
        print(best_estimator)
        with open("best_estimator_SVM","wb") as f:
            pickle.dump(best_estimator,f)
        self.model=best_estimator
        
    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class SVM1wt:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        param_grid  = {'kernel' : ['rbf'],
                        'C' : [0.001, 0.1]
                        ,'class_weight' : [{1.0:10.0, 0.0:1.0}]
                        }
        est = svm.SVC()
        gs_cv = GridSearchCV(est,param_grid).fit(Xtrain,Ytrain)
        print("\n Best Para : ")
        best_params=gs_cv.best_params_
        print(best_params)
        best_estimator=gs_cv.best_estimator_
        print(best_estimator)
        with open("best_estimator_Wt_SVM","wb") as f:
            pickle.dump(best_estimator,f)
        self.model=best_estimator
        
    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class SVM:
    def __init__(self,C):
        self.model=None
        self.prediction=None
        self.C=C
        #self.kernelType=kernelType

    def learn(self,Xtrain,Ytrain):
        #model=svm.SVR(kernel='sigmoid',gamma='auto',C=50)
        #print('\n Kernel : {0}').format(self.kernelType)
        #model=svm.SVC(kernel='linear',C=self.C,class_weight={1.0:1.0,0.0:2.0})
        model=svm.SVC(kernel='rbf',C=self.C)
        model.fit(Xtrain,Ytrain)
        self.model=model
        print('\n Model : {0}').format(self.model)

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class DecisionTreeReg:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        model=DecisionTreeRegressor(max_depth=5)
        model.fit(Xtrain,Ytrain)
        self.model=model

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class RandForest:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        model=RandomForestRegressor(n_estimators=200,max_depth=3)
        model.fit(Xtrain,Ytrain)
        self.model=model

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction



class backPropogation:
    """Back propogation"""
    #
    # Class Members
    #  #
    layerCount=0
    shape=None
    weights=[]

    def __init__(self,layerSize):
        """Initialize the Network"""
        self.layerCount=len(layerSize)-1;
        self.shape=layerSize

        self._layerInput=[]
        self._layerOutput=[]

        # Initialize the weight arrays
        for(l1,l2) in zip(layerSize[:-1],layerSize[1:]):
            """ Example : if layerSize={2,3,2}, then zip will give [(2,3),(3,2)]
            So, weight matrix will be of size [3*3] and [2*4] , as for each input level we need extra bias node"""
            self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))

    def learn(self,input,target,alpha=0.01,regularization=10):
        maxIterations=10000
        threshold=0.01

        for i in range(maxIterations):
            error=self.learnUtil(input, target,alpha)
            #if i%1000==0:
                #print('\n Iteration {0}\t Error : {1:0.6f}').format(i,error)
            if error < threshold:
                #print('\n Convereged after {0} iterations').format(i)
                break

    def learnUtil(self,input,target,learningRate=0.01):
        """
        This method will learn the weights for the different layers.
        """
        delta=[]
        lnCases=input.shape[0]

        # Run the newtwork
        self.run(input)

        #Calulate the error factor and update the weights
        for index in reversed(range(self.layerCount)):
            if index==self.layerCount-1:
            # The last layer just before the output
                output_delta=self._layerOutput[index]-target.T
                squareErr=output_delta**2
                error=np.sum(squareErr)
                delta.append(output_delta*utils.sigm(self._layerInput[index],True))
            else:
            # For other layers
                output_delta=self.weights[index+1].T.dot(delta[-1])
                delta.append(output_delta[:-1,:]*utils.sigm(self._layerInput[index],True))

        # Compute Weight Vector
        for index in range(self.layerCount):
            delta_index=self.layerCount-1-index
            if index==0:
                layerOutput=np.vstack([input.T,np.ones([1,lnCases])])
            else:
                layerOutput=np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index-1].shape[1]])])
            weightedDelta=np.sum(layerOutput[None,:,:].transpose(2,0,1)*delta[delta_index][None,:,:].transpose(2,1,0),axis=0)
            self.weights[index]-=learningRate*weightedDelta

        return error
    def predict(self,testSet,threshold=None):
        threshold=0.5
        Output=self.run(testSet)

        OutputList=list(itertools.chain.from_iterable(Output.tolist()))

        #print('\n Final result : \n Input : {0}\n Output : {1}').format(testSet,OutputList)
        newOutList=[]
        for item in OutputList:
            if item >=threshold:
                newOutList.append(1)
            else:
                newOutList.append(0)

        return newOutList


    def run(self,input):
        """
        This method will run the network for given input and return the output.
        """
        lnCases=input.shape[0]

        #Clear out the previous run network data
        self._layerInput=[]
        self._layerOutput=[]

        #Run the network
        for index in range(self.layerCount):
            if index==0:
                #Input
                layerInput=self.weights[0].dot(np.vstack([input.T,np.ones([1,lnCases])]))
            else:
                #Intermediate layers
                layerInput=self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,self._layerOutput[-1].shape[1]])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(utils.sigm(layerInput))
        return self._layerOutput[-1].T





