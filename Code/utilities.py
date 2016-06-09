import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

"""
    Name : Aniket Gaikwad
    Desc : This is uitility class with all utility functions.
           This includes the functions for parameter evalution, score calculation etc.
"""

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1) 

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))


def getaccuracy(ytest, predictions):
    return accuracy_score(ytest, predictions)


def getRecall(ytest,predictions):
    return recall_score(ytest,predictions)

def getPrecision(ytest,predictions):
    return precision_score(ytest,predictions)

def getF5(precision,recall):
    return (26)*precision*recall / ((25*precision) + recall)

def getAUCROCPlotPoints(ytest,predictions):
    fpr, tpr,thresh= metrics.roc_curve(ytest, predictions, pos_label=1)
    roc_auc=metrics.auc(fpr, tpr)
    return (fpr,tpr,roc_auc)

def getAUCROC(ytest,predictions):
    return roc_auc_score(ytest,predictions)

def fscore(ytest,predictions):
    return metrics.f1_score(ytest,predictions)

def sigm(xvec,Derivative=False):
    """
    Author : Aniket Gaikwad
    Desc: Just combining 'sigmoid' and 'dsigmoid' function
    """
    if not Derivative:
        xvec[xvec < -100] = -100
        return 1.0 / (1.0 + np.exp(np.negative(xvec)))
    else:
        vecsig = sigm(xvec)
        return vecsig * (1 - vecsig)
