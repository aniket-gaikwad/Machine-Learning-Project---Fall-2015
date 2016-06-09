import pandas as pd
import numpy as np
import Statistics_util as Statistics_util
import os

"""
	Name : Aniket Gaikwad
	Desc : Get statistics about dataset.
"""
def getCorrelationMatrix():
	dirPath = 'D:\\Fall 2015\\Machine Learning\\Project\\B555-Project\\MODIFICATIONS'
	fileName = os.path.join(dirPath,'training.csv')
	corr = Statistics_util.loadData(fileName).corr()
	Statistics_util.writeDataframe(corr,'Cor_matrix.csv')


if __name__=="__main__":
	# 1. Get correlation Matrix
	getCorrelationMatrix()