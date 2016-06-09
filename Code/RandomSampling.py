import numpy as np

class randomSampling:
    """
        Name : Aniket Gaikwad
        Desc : Ramdom sampling of data. Set zeroCount and oneCount for proportion of sampling.
    """
    def __init__(self):
        self.model=None
        self.dataset=None

    def getRandomSample(self,X,Y):
        zeroCount=9000
        oneCount=9000
        indexList=[]
        zero_rows = np.random.choice(X.loc[Y==0].index.values, zeroCount)
        one_rows = np.random.choice(X.loc[Y==1].index.values, oneCount)
        sampled_X_0 = X.ix[zero_rows]
        sampled_Y_0 = Y.ix[zero_rows]
        sampled_X_1 = X.ix[one_rows]
        sampled_Y_1 = Y.ix[one_rows]

        return (sampled_X_0.append(sampled_X_1),sampled_Y_0.append(sampled_Y_1))
        