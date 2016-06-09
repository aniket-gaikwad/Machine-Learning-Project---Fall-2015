import  numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import Imputer
import algorithms as algs
import sys
import pickle
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.base import TransformerMixin
import math
    
"""
    
    Name : Aniket Gaikwad
    Description : This File will generate new dataset based on analysis done beforehand.
    Execution : python create_dataset.py h -trainFileName -testFileName
    
"""

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class dataLoad:
    '''
    Author : Aniket Gaikwad
    Desc : This class will produce the dataset with feature selection and elimination.
            This Class contains methods for manual feature selection and Gradient 
            based feature selection.
    '''
    def __init__(self):
        self.trainHeader=None
        self.testHeader=None
        self.RefId=None
        self.prediction=None
    

    ## Utility Functions

    def pickelFile(self,fileName,dataset):
        with open(fileName,"wb") as f:
            pickle.dump(dataset,f)

    
    def loadFile(self,fileName):
        with open(fileName,"rb") as f:
            d=pickle.load(f)
        return d


    def writeDataframe(self,dataFrame,fileName):
        dataFrame.to_csv(fileName+'.csv',sep=',')

            
    def roundOfArray(self,data):
        arr=[]
        print('\n Size of data before roundOfArray : {0}').format(data.shape)
        for row in data:
            result = [round(x,2) for x in row]
            arr.append(result)
        print('\n Size of data before roundOfArray : {0}').format(arr.shape)
        return np.asarray(arr)

    
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
    

    def loadcsv(self,fileName,fileType=None):
        dataset = pd.read_csv(fileName,sep=',',header=0)
        return dataset


    def loadData(self,fileName,fileType=None):
        dataset=None
        print('\n Loading data .....')
        dataset=self.loadcsv(fileName,fileType)
        print('\n Size of data : {0}').format(dataset.shape)
        return dataset


    def removeFeatures(self,dataset,featureRemove,fileType=None):
        df = pd.DataFrame(dataset)
        df.drop(featureRemove,axis=1,inplace=True)
        return df


    def replaceNull(self,dataset):
        return DataFrameImputer().fit_transform(dataset)

    def replaceNewCategory(self,trainDataset,testDataset,categoricalCols):
        for category in categoricalCols:
            uniqTrainCategory = trainDataset[category].unique()
            uniqTestCategory = testDataset[category].unique()
            new_values=list(set(uniqTestCategory)-set(uniqTrainCategory))
            testDataset[category]=self.replaceNewValuesByNan(testDataset[category],category,new_values)
        return testDataset

    def replaceNewValuesByNan(self,data,category,new_values):
        for new_val in new_values:
            data.replace(new_val,np.nan,inplace=True)
        return data

    def replaceCategoriesByNumbers(self,dataset,categoricalCols,fileType):
        
        # 2D Numpy Array of categorical Values
        train_categorial_values=np.array(dataset[categoricalCols])
        
        if(fileType=='Train'):
            # Apply Label Encoder to first Categorical Column
            enc_label = LabelEncoder()
            model = enc_label.fit(train_categorial_values[:,0])
            self.pickelFile(categoricalCols[0]+".pkl",model)
            train_data = model.transform(train_categorial_values[:,0])

            # do the others
            for i in range(1,train_categorial_values.shape[1]):
                enc = LabelEncoder()
                model = enc.fit(train_categorial_values[:,i])
                self.pickelFile(categoricalCols[i]+".pkl",model)
                temp = model.transform(train_categorial_values[:,i])
                train_data = np.column_stack((train_data,temp))
        else:
            # Test file : Load pickle files from disk and apply transformation
            # Apply Label Encoder to first Categorical Column
            enc_label = LabelEncoder()
            model = self.loadFile(categoricalCols[0]+".pkl")
            train_data = model.transform(train_categorial_values[:,0])

            # do the others
            for i in range(1,train_categorial_values.shape[1]):
                enc = LabelEncoder()
                model = self.loadFile(categoricalCols[i]+".pkl")
                temp = model.transform(train_categorial_values[:,i])
                train_data = np.column_stack((train_data,temp))
        
        train_categorial_values = train_data.astype(float)
        train_categorial_values = pd.DataFrame(train_categorial_values,columns=categoricalCols)
        
        # Remove categorical feature and replace with new values
        dataset = self.removeFeatures(dataset,categoricalCols,fileType)
        print('\n size of {0} : {1}').format('dataset',dataset.shape)
        
        # Concatinate categorical and non categorical data
        dataset = pd.concat([train_categorial_values,dataset],axis=1)
        print('\n size of {0} : {1}').format('train_categorial_values',train_categorial_values.shape)
        
        print('\n size after concatination : {0}').format(dataset.shape)
        return dataset

    

    def handleCategoricalVal(self,dataset,categoricalCols,fileType):
        # Create dataframe with first row of dataset as column names
        # 2D Numpy Array of categorical Values
        categorial_values=np.array(dataset[categoricalCols])
        
        model = None
        if(fileType=='Train'):
        # Apply OneHotEncoder to train data
            enc = OneHotEncoder(n_values=40)
            model = enc.fit(categorial_values)
            self.pickelFile("OneHotEncoder.pkl",model)
            cat_data = model.transform(categorial_values)
        else:
        # Apply OneHotEncoder to test data
            enc_label = LabelEncoder()
            model = self.loadFile("OneHotEncoder.pkl")
            cat_data = model.transform(categorial_values)
            

        # Split column headers for Categorical Columns
        # Example : Col_x has 2 values and Col_y has 3 values
        # Generated Columns : Col_x_1, Col_x_2, Col_y_1, Col_y_2, Col_y_3
        cols = [categoricalCols[i] + '_' + str(j) for i in range(0,len(categoricalCols)) \
                for j in range(0,model.n_values_[i]) ]
        cat_data_df = pd.DataFrame(cat_data.toarray(),columns=cols)

        # Remove categorical feature and replace with new values
        dataset = self.removeFeatures(dataset,categoricalCols,fileType)
        print('\n size of {0} : {1}').format('dataset',dataset.shape)
            
        # Concatinate categorical and non categorical data
        dataset = pd.concat([cat_data_df,dataset],axis=1)
        print('\n size of {0} : {1}').format('cat_data_df',cat_data_df.shape)
        print('\n size after concatination : {0}').format(dataset.shape)
        
        return dataset
    


    def replaceCategoriesByLoglikelihood(self,dataset,categoricalCols,fileType):
        # Replace other categorical variables by log likelihood ration
        # log(p(red|kick)) - log(p(red/nonkick))
    
        if(fileType == 'Train'):
            categoricalCols.append('IsBadBuy')
            df = dataset[categoricalCols]
            groupedModel = df.groupby('IsBadBuy')
            categoricalCols.remove('IsBadBuy')
            for feature in categoricalCols:
                d = dict(zip(df[feature].unique(),np.zeros(len(df[feature].unique()))))
                for key,group in groupedModel:
                    newdf = group[['IsBadBuy',feature]]
                    model_prob = newdf.groupby(feature).agg(['count'])/len(newdf)
                    for idx in model_prob.index:
                        if key==0:
                            d[idx]-=math.log(model_prob.loc[idx])
                        else:
                            d[idx]+=math.log(model_prob.loc[idx])
                
                # store probabilities in pkl file
                self.pickelFile(feature+".pkl",d)
                for key,value in d.iteritems():
                    df.loc[df[feature]==key,feature]=value
        else:
            # For Test file load pkl file of Probability for feature & update probability
            df = dataset[categoricalCols]
            for feature in categoricalCols:
                d = self.loadFile(feature+".pkl")
                for key,value in d.iteritems():
                    df.loc[df[feature]==key,feature]=value

        # Remove categorical feature and replace with new values
        dataset = self.removeFeatures(dataset,categoricalCols,fileType)
        print('\n size of {0} : {1}').format('dataset',dataset.shape)

        # Concatinate categorical and non categorical data
        dataset = pd.concat([df[categoricalCols],dataset],axis=1)
        print('\n size of {0} : {1}').format('train_categorial_values',df[categoricalCols].shape)

        print('\n size after concatination : {0}').format(dataset.shape)
        return dataset

    

    def normalizeContinousFeatures(self,dataset,categoricalCols):
        df = dataset[categoricalCols]
        df = df.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
        
        # Remove continous feature and replace with new values
        dataset = self.removeFeatures(dataset,categoricalCols)
        print('\n size of {0} : {1}').format('dataset',dataset.shape)

        # Concatinate categorical and non categorical data
        dataset = pd.concat([df,dataset],axis=1)
        print('\n size of {0} : {1}').format('df',df.shape)

        print('\n size after concatination : {0}').format(dataset.shape)
        return dataset


    
    def handPickedDataLoad(self,trainDataset,testDataset):
        """
            Steps : For both train and test file
            1. Replace categorical values by number
            2. Replace Null values
            3. Remove not useful features
            4. Replace categorical values by binarizing
            5. Replace categorical features by log likelihood
            6. Normalize continous features
        """

        ## These categorical features will be binarized
        categoricalCols = ['Auction','VehYear','Make','Color','Transmission','WheelType',\
                        'Nationality','Size','TopThreeAmericanName','VNST']

        ## These categorical features will be replaced by Log-likelihood
        categoricalCols1 = ['Model','Trim','SubModel']

        ## Continous values will be normalized
        continousCols=['VehOdo','MMRAcquisitionAuctionAveragePrice',\
              'MMRAcquisitionAuctionCleanPrice','MMRAcquisitionRetailAveragePrice',\
              'MMRAcquisitonRetailCleanPrice','MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice', \
              'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice','WarrantyCost']

        ## Features to remove
        featureRemove = ['RefId','WheelTypeID','PurchDate','PRIMEUNIT',\
                         'AUCGUART','VNZIP1','BYRNO']      
        
        
        ## 1. Remove not useful features
        trainDataset=self.removeFeatures(trainDataset,featureRemove,'Train')
        print('\n Size of trainDataset after removing features: {0}').format(trainDataset.shape)
        testDataset=self.removeFeatures(testDataset,featureRemove,'Test')
        print('\n Size of testDataset after removing features: {0}').format(testDataset.shape)

        ## 2. Replace categorical values by number
        ## 3. Replace Null values
        testDataset = self.replaceNewCategory(trainDataset,testDataset,categoricalCols)
        trainDataset=self.replaceNull(trainDataset)
        print('\n Size of trainDataset after null replacement: {0}').format(trainDataset.shape)
        testDataset=self.replaceNull(testDataset)
        print('\n Size of testDataset after null replacement: {0}').format(testDataset.shape)
        trainDataset = self.replaceCategoriesByNumbers(trainDataset,categoricalCols,'Train')
        print('\n Size of trainDataset after replaceCategoriesByNumbers : {0}').format(trainDataset.shape)
        testDataset = self.replaceCategoriesByNumbers(testDataset,categoricalCols,'Test')
        print('\n Size of testDataset after replaceCategoriesByNumbers : {0}').format(testDataset.shape)

        ## 4. Split categorical values by binarizing
        trainDataset = self.handleCategoricalVal(trainDataset,categoricalCols,'Train')
        print('\n Size of trainDataset after handleCategoricalVal : {0}').format(trainDataset.shape)
        testDataset = self.handleCategoricalVal(testDataset,categoricalCols,'Test')
        print('\n Size of testDataset after handleCategoricalVal : {0}').format(testDataset.shape)
        
        ## 5. Replace categorical features by log likelihood
        trainDataset = self.replaceCategoriesByLoglikelihood(trainDataset,categoricalCols1,'Train')
        print('\n Size of trainDataset after replaceCategoriesByLoglikelihood : {0}').format(trainDataset.shape)
        testDataset = self.replaceCategoriesByLoglikelihood(testDataset,categoricalCols1,'Test')
        print('\n Size of testDataset after replaceCategoriesByLoglikelihood : {0}').format(testDataset.shape)
        
        ##  6. Normalize continous features
        trainDataset = self.normalizeContinousFeatures(trainDataset,continousCols)
        print('\n Size of trainDataset after normalizeContinousFeatures : {0}').format(trainDataset.shape)
        testDataset = self.normalizeContinousFeatures(testDataset,continousCols)
        print('\n Size of testDataset after normalizeContinousFeatures : {0}').format(testDataset.shape)

        return (trainDataset,testDataset)



if __name__=="__main__":
    if(len(sys.argv)!=4):
        print("\n Invalid Number of arguments ")
        print("\n Please use following format for running...")
        print("\n python create_dataset.py [HhGg] -trainFileName -testFileName")
        sys.exit(0)
    inputTrainFile=sys.argv[2]
    inputTestFile=sys.argv[3]
    trainFileName="DefaultTrain"
    testFileName="DefaultTest"

    if(sys.argv[1]=="H" or sys.argv[1]=="h"):
        trainFileName="HandPickedFeaturesTrain"
        testFileName="HandPickedFeaturesTest"

    # Load File
    dataLoadObj=dataLoad()
    trainDataset=dataLoadObj.loadData(inputTrainFile,'train')
    testDataset=dataLoadObj.loadData(inputTestFile,'test')

    if(sys.argv[1]=="H" or sys.argv[1]=="h"):
        (trainDataset,testDataset)=dataLoadObj.handPickedDataLoad(trainDataset,testDataset)
        dataLoadObj.writeDataframe(trainDataset,trainFileName)
        dataLoadObj.writeDataframe(testDataset,testFileName)

   

    