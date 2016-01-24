import  numpy as np
import csv
#import DataVisualization as DV
import  pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import algorithms as algs
import utilities as util
from scipy.stats import ttest_ind
from sklearn import metrics


class Kfold:
    def __init__(self,No_of_Folds=None):
        self.No_of_Folds=No_of_Folds
        self.kf=None

    def getFoldIndices(self,dataset):
        kf = KFold(dataset.shape[0],n_folds=self.No_of_Folds)
        self.kf=kf
        return kf

    def runFolds(self,dataset):
        i=1
        FoldAccuracy={}
        for train_index, test_index in self.kf:
            print('\n For Fold : {0}').format(i)
            train, test = dataset[train_index], dataset[test_index]
            print('\n Train : {0} \n Test : {1}').format(train.shape, test.shape)
            numinputs = train.shape[1]-1
            Xtrain = train[:,0:numinputs]
            Ytrain = train[:,numinputs]
            Xtest = test[:,0:numinputs]
            Ytest = test[:,numinputs]
            print('\n XTrain : {0} \n YTrain : {1} \n Xtest : {2} \n YTest : {3}').format(Xtrain.shape,Ytrain.shape,Xtest.shape,Ytest.shape)
            ### Run the classifiers
            classalgs = {#'Logistic Regression_10' : algs.LogisticRegression(C=10),
                        #'Logistic Regression_1' : algs.LogisticRegression(C=1),
                        'Logistic Regression_.1' : algs.LogisticRegression(C=0.1),
                        #'Logistic Regression_.01' : algs.LogisticRegression(C=0.01),
                        #'Logistic Regression_.001' : algs.LogisticRegression(C=0.001),
                         #'GradientBoostingClassifier_10' : algs.GradientBoost(n_estimators=10),
                         #'GradientBoostingClassifier_5' : algs.GradientBoost(n_estimators=5),
                         #'GradientBoostingClassifier_4' : algs.GradientBoost(n_estimators=4),
                         #'GradientBoostingClassifier_3' : algs.GradientBoost(n_estimators=3),
                         #'GradientBoostingClassifier_2' : algs.GradientBoost(n_estimators=2),
                         #'Gauassian SVM_300' :algs.SVM(C=300),
                         #'Gauassian SVM_200' :algs.SVM(C=200),
                         'Gauassian SVM_100' :algs.SVM(C=100),
                         #'Gauassian SVM_50' :algs.SVM(C=50),
                         #'Gauassian SVM_20' :algs.SVM(C=20),
                         #'Neural Net_4':algs.backPropogation((Xtrain.shape[1],4,1)),
                         #'Neural Net_8':algs.backPropogation((Xtrain.shape[1],8,1)),
                         #'Neural Net_16':algs.backPropogation((Xtrain.shape[1],16,1)),
                         #'Neural Net_32':algs.backPropogation((Xtrain.shape[1],32,1)),
                         #'Neural Net_64':algs.backPropogation((Xtrain.shape[1],64,1))
                          }

            accuracyD={}
            for learnername, learner in classalgs.iteritems():
                print 'Running learner = ' + learnername
                learner.learn(Xtrain, Ytrain)
                predictions = learner.predict(Xtest)
                recall = util.getRecall(Ytest, predictions)
                print '\n Recall for ' + learnername + ': ' + str(recall)
                AUCROC =util.getAUCROC(Ytest, predictions)
                print '\n AUC ROC Score for ' + learnername + ': ' + str(AUCROC)
                AUCROCPlotPoints =util.getAUCROCPlotPoints(Ytest, predictions)
                print('\n tpr : {0} fpr : {1} auc_roc : {2} learnername : {3}').format(AUCROCPlotPoints[0],AUCROCPlotPoints[1],AUCROCPlotPoints[2],learnername)
                self.plotGraph(AUCROCPlotPoints,learnername)
                f1_score = util.fscore(Ytest, predictions)
                print '\n f1_score for ' + learnername + ': ' + str(f1_score)
                accuracy = util.getaccuracy(Ytest, predictions)
                print 'Accuracy for ' + learnername + ': ' + str(accuracy)
                accuracyD[learnername]=AUCROC
            FoldAccuracy[i]=accuracyD
            i=i+1
        self.StatisticalSignificance(FoldAccuracy)

    def plotGraph(self,AUCROCPlotPoints,learnerName):
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

    def StatisticalSignificance(self,FoldAccuracy):
        print('\n ***************** Final Result *************\n {0}').format(FoldAccuracy)
        MasterL2=[]
        #['L2_1','L2_2','L2_3','L2_4','L2_5']
        L2_1=[]   ## _10
        L2_2=[]   ## _1
        L2_3=[]   ## _0.1
        L2_4=[]   ## _0.01
        L2_5=[]   ## _0.001

        MasterGB=[]
        #['GB_1','GB_2','GB_3','GB_4','GB_5']
        GB_1=[]   ## _10
        GB_2=[]   ## _5
        GB_3=[]   ## _4
        GB_4=[]   ## _3
        GB_5=[]   ## _2

        MasterSVM=[]
        #['SVM_1','SVM_2','SVM_3','SVM_4','SVM_5']
        SVM_1=[]   ## _300
        SVM_2=[]   ## _200
        SVM_3=[]   ## _100
        SVM_4=[]   ## _50
        SVM_5=[]   ## _20

        MasterNN=[]
        #['NN_1','NN_2','NN_3','NN_4','NN_5']
        NN_1=[]   ## _4
        NN_2=[]   ## _8
        NN_3=[]   ## _16
        NN_4=[]   ## _32
        NN_5=[]   ## _64

        for fold_key,fold_dict in FoldAccuracy.iteritems():
            for classifier,result in fold_dict.iteritems():
                if classifier=='GradientBoostingClassifier_10':
                    GB_1.append(result)
                if classifier=='GradientBoostingClassifier_5':
                    GB_2.append(result)
                if classifier=='GradientBoostingClassifier_4':
                    GB_3.append(result)
                if classifier=='GradientBoostingClassifier_3':
                    GB_4.append(result)
                if classifier=='GradientBoostingClassifier_2':
                    GB_5.append(result)

                if classifier=='L2-Logistic Regression_10':
                    L2_1.append(result)
                if classifier=='L2-Logistic Regression_1':
                    L2_2.append(result)
                if classifier=='L2-Logistic Regression_.1':
                    L2_3.append(result)
                if classifier=='L2-Logistic Regression_.01':
                    L2_4.append(result)
                if classifier=='L2-Logistic Regression_.001':
                    L2_5.append(result)

                if classifier=='Gauassian SVM_300':
                    SVM_1.append(result)
                if classifier=='Gauassian SVM_200':
                    SVM_2.append(result)
                if classifier=='Gauassian SVM_100':
                    SVM_3.append(result)
                if classifier=='Gauassian SVM_50':
                    SVM_4.append(result)
                if classifier=='Gauassian SVM_20':
                    SVM_5.append(result)

                if classifier=='Neural Net_4':
                    NN_1.append(result)
                if classifier=='Neural Net_8':
                    NN_2.append(result)
                if classifier=='Neural Net_16':
                    NN_3.append(result)
                if classifier=='Neural Net_32':
                    NN_4.append(result)
                if classifier=='Neural Net_64':
                    NN_5.append(result)
        MasterGB.append(GB_1)
        MasterGB.append(GB_2)
        MasterGB.append(GB_3)
        MasterGB.append(GB_4)
        MasterGB.append(GB_5)
        MasterL2.append(L2_1)
        MasterL2.append(L2_2)
        MasterL2.append(L2_3)
        MasterL2.append(L2_4)
        MasterL2.append(L2_5)
        MasterNN.append(NN_1)
        MasterNN.append(NN_2)
        MasterNN.append(NN_3)
        MasterNN.append(NN_4)
        MasterNN.append(NN_5)
        MasterSVM.append(SVM_1)
        MasterSVM.append(SVM_2)
        MasterSVM.append(SVM_3)
        MasterSVM.append(SVM_4)
        MasterSVM.append(SVM_5)
        ### Find mean for all variants of Gradient Boost
        L1=[np.mean(np.asarray(GB_1)),np.mean(np.asarray(GB_2)),np.mean(np.asarray(GB_3)),np.mean(np.asarray(GB_4)),np.mean(np.asarray(GB_5))]
        #print ('\n Mean GB : {0}').format(L1)
        print('\n Max Gradient Boost: {0}').format(L1.index(max(L1)))
        Max_GB=MasterGB[L1.index(max(L1))]
        print('\n Max_GB : {0}').format(Max_GB)

        ### Find mean for all variants of Logistic regression
        L2=[np.mean(np.asarray(L2_1)),np.mean(np.asarray(L2_2)),np.mean(np.asarray(L2_3)),np.mean(np.asarray(L2_4)),np.mean(np.asarray(L2_5))]
        #print ('\n Mean L2 : {0}').format(L2)
        print('\n Max Logistic Regression: {0}').format(L2.index(max(L2)))
        Max_L2=MasterL2[L2.index(max(L2))]
        print('\n Max_L2 : {0}').format(Max_L2)

        ### Find mean for all variants of Neural Net
        L3=[np.mean(np.asarray(NN_1)),np.mean(np.asarray(NN_2)),np.mean(np.asarray(NN_3)),np.mean(np.asarray(NN_4)),np.mean(np.asarray(NN_5))]
        #print ('\n Mean NN : {0}').format(L3)
        print('\n Max Neural Net: {0}').format(L3.index(max(L3)))
        Max_NN=MasterNN[L3.index(max(L3))]
        print('\n Max_NN : {0}').format(Max_NN)

        ### Find mean for all variants of SVM
        L4=[np.mean(np.asarray(SVM_1)),np.mean(np.asarray(SVM_2)),np.mean(np.asarray(SVM_3)),np.mean(np.asarray(SVM_4)),np.mean(np.asarray(SVM_5))]
        #print ('\n Mean SVM : {0}').format(L4)
        print('\n Max SVM: {0}').format(L4.index(max(L4)))
        Max_SVM=MasterSVM[L4.index(max(L4))]
        print('\n Max_SVM : {0}').format(Max_SVM)

        ########################### T-Val and P-Val Calculation ################################
        ## Test Gradient Boost  and Logistic
        pVal=self.getPval(np.asarray(Max_GB),np.asarray(Max_L2))
        print('\n Gradient Boost Vs Logistic Regression')
        print('\n P val : {0}').format(pVal)

        ## Test Gradient Boost  and Neural Net
        pVal=self.getPval(np.asarray(Max_GB),np.asarray(Max_NN))
        print('\n Gradient Boost Vs Neural Net')
        print('\n P val : {0}').format(pVal)

        ## Test Gradient Boost  and SVM
        pVal=self.getPval(np.asarray(Max_GB),np.asarray(Max_SVM))
        print('\n Gradient Boost Vs SVM')
        print('\n P val : {0}').format(pVal)

        ## Test Neural Net and Logistic
        pVal=self.getPval(np.asarray(Max_NN),np.asarray(Max_L2))
        print('\n Neural Net and Logistic')
        print('\n P val : {0}').format(pVal)

        ## Test Neural Net and SVM
        pVal=self.getPval(np.asarray(Max_NN),np.asarray(Max_SVM))
        print('\n Neural Net and SVM')
        print('\n P val : {0}').format(pVal)

        ## Test Logistic  and SVM
        pVal=self.getPval(np.asarray(Max_L2),np.asarray(Max_SVM))
        print('\n Logistic Regression and SVM')
        print('\n P val : {0}').format(pVal)


    def getPval(self,S1,S2):
        return ttest_ind(S1,S2)
