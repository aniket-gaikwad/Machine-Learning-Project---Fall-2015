import Visualize_Util as V_util
import run_util as run_util
import os

"""
    Name : Aniket Gaikwad
    Desc : Initial Data Visualization.
"""

def loadData(runObj):
	dirPath = 'D:\\Fall 2015\\Machine Learning\\Project\\B555-Project\\MODIFICATIONS'
	trainDataset = runObj.loadData(os.path.join(dirPath,'training.csv'),'train')
	return trainDataset

def visualizeAll():
	runObj = run_util.run()
	X = loadData(runObj)
	V_util.visualizeAllFeatures(X)

def visualizeCategorial(featureList):
	runObj = run_util.run()
	X = loadData(runObj)
	X = X[featureList]
	V_util.visualizeCategorial(X)

def visualizeCategoricalClassLabel(featureList):
	runObj = run_util.run()
	X = loadData(runObj)
	V_util.visualizeCategoricalClassLabel(X,featureList)

def visualizePieChart(featureList):
	runObj = run_util.run()
	X = loadData(runObj)
	V_util.visualizePieChart(X,featureList)	

if __name__ == "__main__":

	# 1. All Feature Visualization
	visualizeAll()

	# 2. Categorical Feature Visualization
	featureList= ['Auction','VehYear','Make','Color','Transmission','WheelType',\
                        'Nationality','Size','TopThreeAmericanName','VNST','Model','Trim','SubModel']
	visualizeCategorial(featureList)

	# 3. Categorical features wrt to class label
	visualizeCategoricalClassLabel(featureList)

	# 4. Pie charts
	featureList= ['IsBadBuy','Auction','PRIMEUNIT','AUCGUART','IsOnlineSale']
	visualizePieChart(featureList)

	
