import pandas as pd
import pickle
"""
	Name : Aniket Gaikwad
	Desc : Utility functions for Statistics.
"""

def loadcsv(fileName,fileType=None):
        dataset = pd.read_csv(fileName,sep=',',header=0)
        return dataset


def loadData(fileName,fileType=None):
    dataset=None
    print('\n Loading data .....')
    dataset=loadcsv(fileName,fileType)
    print('\n Size of data : {0}').format(dataset.shape)
    return dataset

def pickelFile(fileName,dataset):
        with open(fileName,"wb") as f:
            pickle.dump(dataset,f)

    
def loadFile(fileName):
    with open(fileName,"rb") as f:
        d=pickle.load(f)
    return d

def writeDataframe(dataFrame,fileName):
	dataFrame.to_csv(fileName+'.csv',sep=',')
