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
import KfoldCrossValidation as Kfold
import RandomSampling as RS

class dataLoad:
    def __init__(self):
        self.header=None
        self.RefId=None
        self.prediction=None



    def splitdataset(self,dataset, trainsize=6600, testsize=3300, testfile=None):
        randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
        ## Generate the random indices bewteen 0-number of rows with size of random array as 800
        numinputs = dataset.shape[1]-1
        Xtrain = dataset[randindices[0:trainsize],0:numinputs]
        ytrain = dataset[randindices[0:trainsize],numinputs]
        Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
        ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

        if testfile is not None:
            testdataset = self.loadcsv(testfile)
            Xtest = dataset[:,0:numinputs]
            ytest = dataset[:,numinputs]

        # Add a column of ones; done after to avoid modifying entire dataset

        #Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
        #Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))

        return ((Xtrain,ytrain), (Xtest,ytest))

    def featureSel(self,dataset,fileType=None):
        header=self.header
        featureRemove=['PurchMonth','PurchDay','PurchYear','PRIMEUNIT','AUCGUART','VNZIP1','SubModel','Model','BYRNO']
        indexRemove=[header.index(x) for x in featureRemove]
        #print('\n Index Remove : {0}').format(indexRemove)

        dataset=np.delete(dataset,indexRemove,axis=1)
        head=np.asarray(self.header)
        head=np.delete(head,indexRemove)
        self.header=head.tolist()

        return dataset

    def loadcsv(self,fileName,fileType=None):
        with open(fileName, 'r') as f:
            header = f.readline()
        header=header.split(',')
        headerList=[]
        for x in header:
            headerList.append(x)
        ## Replace '\n' by ''
        headerList=[item.rstrip('\n') for item in headerList]
        headerList=[item.rstrip('\r') for item in headerList]
        #### For test file remove the 'RefId'
        #if fileType=="test":
        #    headerList.pop(0)

        self.header=headerList
        #print('\n headerDict : {0}').format(headerList)
        dataset=np.genfromtxt(fileName, delimiter=',')
        ### Remove the header
        dataset=dataset[1:,:]
        ### Swap the label column to last of dataset ONLY FOR TRAIN FILE
        if fileType=="train":
            dataset[:,[1,-1]]=dataset[:,[-1,1]]
        ### For test file Just store RefId in variable
        if fileType=="test":
            self.RefId=dataset[:-6,0]
            print('Ref : {0}').format(self.RefId)
        ### Feature selection
        dataset=self.featureSel(dataset)
        return dataset




    def loadData(self,fileName,fileType=None):
        dataset=None
        print('\n Loading data .....')
        dataset=self.loadcsv(fileName,fileType)
        print('\n Size of data : {0}').format(dataset.shape)
        #print('\n class label : {0}').format(dataset[:,-1])
        dataset[np.isnan(dataset)]=100
        dataset=dataset.astype(int)
        #dataset=dataset[0:10000,:]
        ### Sepearate the label and features
        numinputs=dataset.shape[1]-1
        header=self.header
        featureSplit=['Auction','VehYear','Make','Color','Transmission','WheelTypeID','Nationality','Size','TopThreeAmericanName',\
                      'VNST']
        indexSelect=[header.index(x) for x in featureSplit]
        #print('\n Index Select : {0}').format(indexSelect)

        featureNotSplit=['MMRAcquisitionAuctionCleanPrice','MMRCurrentRetailAveragePrice','MMRAcquisitionRetailAveragePrice',\
                         'IsOnlineSale','VehOdo','VehicleAge','MMRCurrentRetailCleanPrice','MMRAcquisitionAuctionAveragePrice',\
                         'MMRAcquisitonRetailCleanPrice','MMRCurrentAuctionAveragePrice','VehBCost','Trim',\
                         'MMRCurrentAuctionCleanPrice','WarrantyCost']
        indexNotSelect=[header.index(x) for x in featureNotSplit]
        #print('\n Index Not Select : {0}').format(indexNotSelect)
        if fileType=="train":
            XdatasetSplit=dataset[:,indexSelect]
            #print('\n XdatasetSplit : {0}').format(XdatasetSplit.shape)
            Xdataset=dataset[:,indexNotSelect]
            #print('\n Xdataset : {0}').format(Xdataset.shape)
            Ydataset=dataset[:,numinputs:]
        else:
            XdatasetSplit=dataset[:,indexSelect]
            Xdataset=dataset[:,indexNotSelect]
        ####################
        print('\n Dataset Transform Starts ........')

        enc = OneHotEncoder()
        enc.fit(XdatasetSplit)
        dataset=enc.transform(XdatasetSplit).toarray()
        #print('\n Size after OneHotEncoder : {0}').format(dataset.shape)
        dataset=np.append(Xdataset,dataset,axis=1)
        print('\n Size after OneHotEncoder : {0}').format(dataset.shape)
        #print('\n Ydataset : {0}').format(Ydataset)
        if fileType=="train":
            dataset=np.append(dataset,Ydataset,axis=1)

        return dataset


    def runClassifiers(self,dataset,testDataset):
        #trainset, testset = self.splitdataset(dataset)
        numinputs = dataset.shape[1]-1
        Xtrain = dataset[:,0:numinputs]
        ytrain = dataset[:,numinputs]
        print('Split into train={0} and test={1} ').format(Xtrain.shape, testDataset.shape)
        classalgs = {   'Logistic Regression_10' : algs.LogisticRegression(C=10),
                        'Logistic Regression_1' : algs.LogisticRegression(C=1),
                        'Logistic Regression_.1' : algs.LogisticRegression(C=0.1),
                        'Logistic Regression_.01' : algs.LogisticRegression(C=0.01),
                        'Logistic Regression_.001' : algs.LogisticRegression(C=0.001),
                        #'GradientBoostingClassifier_10' : algs.GradientBoost(n_estimators=10),
                         #'GradientBoostingClassifier_5' : algs.GradientBoost(n_estimators=5),
                         #'GradientBoostingClassifier_4' : algs.GradientBoost(n_estimators=4),
                         #'GradientBoostingClassifier_3' : algs.GradientBoost(n_estimators=3),
                         #'GradientBoostingClassifier_2' : algs.GradientBoost(n_estimators=2),
                         'Gauassian SVM_300' :algs.SVM(C=300),
                         'Gauassian SVM_200' :algs.SVM(C=200),
                         'Gauassian SVM_100' :algs.SVM(C=100),
                         'Gauassian SVM_50' :algs.SVM(C=50),
                         'Gauassian SVM_20' :algs.SVM(C=20),
                         #'Neural Net_4':algs.backPropogation((Xtrain.shape[1],4,1)),
                         #'Neural Net_8':algs.backPropogation((Xtrain.shape[1],8,1)),
                         #'Neural Net_16':algs.backPropogation((Xtrain.shape[1],16,1)),
                         #'Neural Net_32':algs.backPropogation((Xtrain.shape[1],32,1)),
                         #'Neural Net_64':algs.backPropogation((Xtrain.shape[1],64,1))
                         }

        # Runs all the algorithms on the data and print out results
        for learnername, learner in classalgs.iteritems():
            print 'Running learner = ' + learnername
            # Train model
            #learner.featureSelection(trainset[0])
            learner.learn(Xtrain, ytrain)
            # Test model
            predictions = learner.predict(testDataset)
            self.prediction=predictions
            print predictions
            #accuracy = util.getaccuracy(testset[1], predictions)
            #print 'Accuracy for ' + learnername + ': ' + str(accuracy)
            fileName='output_'+learnername+'.csv'
            self.writeFile(fileName)

    def writeFile(self,fileName):
        print('\n Writting starts : ')
        data=[]
        RefId=self.RefId
        RefId=RefId.astype(int)
        predictions=self.prediction
        predictions=predictions.astype(int)
        print('\n RefId : {0}').format(RefId.shape[0])
        print('\n predictions : {0}').format(predictions.shape[0])
        for i in range(RefId.shape[0]):
            data.append([RefId[i],predictions[i]])
        print('\n Data : \n {0}').format(data)
        with open(fileName, 'wb') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerows(data)



if __name__=="__main__":
    fileName='updatedTraining.csv'
    dataLoadObj=dataLoad()
    Traindataset=dataLoadObj.loadData(fileName,'train')
    ############ Random Sampling ###################
    RSObj=RS.randomSampling()
    Traindataset=RSObj.getRandomSample(Traindataset)
    print('\n Train : {0}').format(Traindataset.shape)
    '''
    fileName='updatetest.csv'
    testDataset=dataLoadObj.loadData(fileName,'test')
    testDataset=testDataset[0:-6]
    print('\n Test : {0}').format(testDataset.shape)
    dataLoadObj.runClassifiers(Traindataset,testDataset)

    '''
    print('\n Cross Validation : ')
    randindices = np.random.randint(0,Traindataset.shape[0],10000)
    Traindataset=Traindataset[randindices,:]
    Kobj=Kfold.Kfold(No_of_Folds=10)
    kf=Kobj.getFoldIndices(Traindataset)
    Kobj.runFolds(Traindataset)




