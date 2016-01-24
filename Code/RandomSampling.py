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


class randomSampling:
    def __init__(self):
        self.model=None
        self.dataset=None

    def getRandomSample(self,dataset):
        zeroCount=18000
        oneCount=9000
        indexList=[]
        for i in range(dataset.shape[0]):
            if zeroCount<=0 and oneCount<=0:
                break
            if dataset[i][dataset.shape[1]-1]==0 and zeroCount>0:
                zeroCount-=1
                indexList.append(i)
            if dataset[i][dataset.shape[1]-1]==1 and oneCount >0:
                oneCount-=1
                indexList.append(i)
        dataset=dataset[indexList,:]
        print('\n dataset : {0}').format(dataset.shape)
        return dataset

