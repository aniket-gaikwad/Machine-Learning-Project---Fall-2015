import  numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
import algorithms as algs
import sys
import pickle

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
    
    def loadcsv(self,fileName,fileType=None,classLabelColumnNo=-1):
        '''
         This function will take input as fileName,fileType and location of class label column.
         When filetype is train, it will swap last column with class label column so that class
         label will be a last column in train data.
        '''
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
        dataset=np.genfromtxt(fileName, delimiter=',')
        ### Remove the header
        dataset=dataset[1:,:]
        ### Swap the label column to last of dataset ONLY FOR TRAIN FILE
        if fileType=="train":
            dataset[:,[classLabelColumnNo,-1]]=dataset[:,[-1,classLabelColumnNo]]
            headerList[int(classLabelColumnNo)],headerList[-1]=headerList[-1],headerList[int(classLabelColumnNo)]
            self.trainHeader=headerList[:]
        if fileType=="test":
            self.testHeader=headerList[:]
        return dataset


    def loadData(self,fileName,fileType=None,classLabelColumnNo=-1):
        '''
            Load csv file, replace null values by median.
        '''
        dataset=None
        print('\n Loading data .....')
        dataset=self.loadcsv(fileName,fileType,classLabelColumnNo)
        print('\n Size of data : {0}').format(dataset.shape)
        # Null handling
        dataset=self.replaceNull(dataset)
        dataset=self.roundOfArray(dataset)
        print('\n Size of data after null replacement: {0}').format(dataset.shape)
        return dataset

    def replaceNull(self,dataset):
        '''
            Replace null values by 'Median' or 'Mean'
        '''
        df=dataset
        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
        df=imp.fit(df).transform(df)
        self.pickelFile("test",df)
        return df

    def featureSel(self,dataset,featureRemove,fileType):
        '''
            Based on list of features to remove, remove the features from dataset.
        '''
        if(fileType=="train"):
            header=self.trainHeader
        if(fileType=="test"):
            header=self.testHeader
        indexRemove=[header.index(x) for x in featureRemove]
        dataset=np.delete(dataset,indexRemove,axis=1)
        head=np.asarray(header)
        head=np.delete(head,indexRemove)
        if(fileType=="train"):
            self.trainHeader=head.tolist()
        if(fileType=="test"):
            self.testHeader=head.tolist()
        return dataset
    
    def gradientDataLoad(self,fileType=None,dataset=None):
        '''
            Gradient Boosting algorithm based feature selection.
        '''
        featureRemove=['RefId','VNZIP1','BYRNO']
        dataset=self.featureSel(dataset,featureRemove,fileType)
        if(fileType=="train"):
            header=self.trainHeader
            header=header[:-1]
            numinputs = dataset.shape[1]-1
            X = dataset[:,0:numinputs]
            Y = dataset[:,numinputs:]
            algs.GradientBoost(header,100).learn(X,Y[:,0])
        else:
            header=self.testHeader
            X = dataset
        feature_importances_=self.loadFile("feature_importances.pkl")
        feature_importances_=feature_importances_[feature_importances_ > 0].keys().tolist()
        featureRemove=filter(lambda x : x not in feature_importances_, header)
        dataset=self.featureSel(X ,featureRemove,fileType)
        print('\n Train : {0} , Size of data : {1}').format(fileType,dataset.shape)
        if fileType=="train":
            dataset=np.append(dataset,Y,axis=1)
        return dataset

    def handPickedDataLoad(self,fileType=None,dataset=None):
        '''
            Manual Feature selection.
        '''
        featureRemove=['RefId','PurchMonth','PurchDay','PurchYear','PRIMEUNIT',\
        'AUCGUART','VNZIP1','SubModel','Model','BYRNO']
        dataset=self.featureSel(dataset,featureRemove,fileType)
        if(fileType=="train"):
            header=self.trainHeader
        if(fileType=="test"):
            header=self.testHeader
        print('\n Size of data : {0}').format(dataset.shape)
        ### Sepearate the label and features
        numinputs=dataset.shape[1]-1
        #print(header)
        featureSplit=['Auction','VehYear','Make','Color','Transmission','WheelTypeID',\
                        'Nationality','Size','TopThreeAmericanName','VNST']
        indexSelect=[header.index(x) for x in featureSplit]
        #print(indexSelect)
        featureNotSplit=['MMRAcquisitionAuctionCleanPrice','MMRCurrentRetailAveragePrice',\
                            'MMRAcquisitionRetailAveragePrice','IsOnlineSale','VehOdo',\
                            'VehicleAge','MMRCurrentRetailCleanPrice',\
                            'MMRAcquisitionAuctionAveragePrice','MMRAcquisitonRetailCleanPrice',\
                            'MMRCurrentAuctionAveragePrice','VehBCost','Trim',\
                            'MMRCurrentAuctionCleanPrice','WarrantyCost']
        indexNotSelect=[header.index(x) for x in featureNotSplit]
        enc = OneHotEncoder()
        encoder=None
        #dataset=dataset.astype(int)
        if fileType=="train":
            XdatasetSplit=dataset[:,indexSelect]
            Xdataset=dataset[:,indexNotSelect]
            Ydataset=dataset[:,numinputs:]
            encoder=enc.fit(XdatasetSplit)
            self.pickelFile("encoder.pkl",encoder)
        else:
            XdatasetSplit=dataset[:,indexSelect]
            Xdataset=dataset[:,indexNotSelect]
            encoder=self.loadFile("encoder.pkl")
            #encoder.fit(XdatasetSplit)
           
        print('\n Dataset Transform Starts ........')
        dataset=encoder.transform(XdatasetSplit)
        print('\n No of features after OneHotEncoder : {0}').format(dataset.shape)
        dataset=dataset.toarray()
        dataset=np.append(Xdataset,dataset,axis=1)
        print('\n Size after OneHotEncoder : {0}').format(dataset.shape)
        if fileType=="train":
            dataset=np.append(dataset,Ydataset,axis=1)
        return dataset


    def pickelFile(self,fileName,dataset):
        with open(fileName,"wb") as f:
            pickle.dump(dataset,f)

    def loadFile(self,fileName):
        with open(fileName,"rb") as f:
            d=pickle.load(f)
        return d

    def roundOfArray(self,data):
        arr=[]
        for row in data:
            result = [round(x,2) for x in row]
            arr.append(result)
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



if __name__=="__main__":
    if(len(sys.argv)!=5):
        print("\n Invalid Number of arguments ")
        print("\n Please use following format for running...")
        print("\n python create_dataset.py [HhGg] -trainFileName -ClassLabelColumnNo -testFileName")
        sys.exit(0)
    inputTrainFile=sys.argv[2]
    classLabelColumnNo=sys.argv[3]
    inputTestFile=sys.argv[4]
    trainFileName="DefaultTrain.csv"
    testFileName="DefaultTest.csv"

    if(sys.argv[1]=="H" or sys.argv[1]=="h"):
        trainFileName="HandPickedFeaturesTrain"
        testFileName="HandPickedFeaturesTest"
    if(sys.argv[1]=="G" or sys.argv[1]=="g"):
        trainFileName="GradientFeaturesTrain"
        testFileName="GradientFeaturesTest"

    # Load File
    dataLoadObj=dataLoad()
    trainDataset=dataLoadObj.loadData(inputTrainFile,'train',classLabelColumnNo)
    testDataset=dataLoadObj.loadData(inputTestFile,'test')
    if(sys.argv[1]=="H" or sys.argv[1]=="h"):
        trainDataset=dataLoadObj.handPickedDataLoad('train',trainDataset)
        testDataset=dataLoadObj.handPickedDataLoad('test',testDataset)
    if(sys.argv[1]=="G" or sys.argv[1]=="g"):
        trainDataset=dataLoadObj.gradientDataLoad('train',trainDataset)
        testDataset=dataLoadObj.gradientDataLoad('test',testDataset)

    dataLoadObj.pickelFile(trainFileName+".pkl",trainDataset)
    dataLoadObj.pickelFile(testFileName+".pkl",testDataset)

    