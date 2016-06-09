import matplotlib.pyplot as plt

"""
    Name : Aniket Gaikwad
    Desc : Initial Data Visualization Utility functions.
"""

def visualizeAllFeatures(X):
    """
        Visualize all features.
    """
    cols = list(X.columns.values)
    for var in cols:
        try:
            grouped=X.groupby([var])
            k=[]
            g=[]
            for key,group in grouped:
                k.append(key)
                g.append(group[var].count())
            plt.plot(k,g,label=var)
            plt.xlabel(var)
            plt.ylabel('Frequency')
            plt.title('Frequency distribution for : '+var)
            plt.show()
        except:
            continue

def visualizeCategorial(X):
    """ 
        Visualize Categorical features.
    """
    cols = list(X.columns.values)
    for var in cols:
        ax = X[var].value_counts().plot(kind='bar')
        ax.set_xlabel(var)
        ax.set_ylabel('Frequency')
        ax.set_title('Frequency distribution for : '+var)
        plt.show()

def visualizeCategoricalClassLabel(X,cols):
    """
        Visualize features with respect to class labels.
    """
    for var in cols:
        df2 = X.groupby([var, 'IsBadBuy'])[var].count().unstack('IsBadBuy').fillna(0)
        ax = df2.plot(kind='bar',stacked=True)
        ax.set_xlabel(var)
        ax.set_ylabel('Frequency of Class Label')
        ax.set_title('Distribution of class label w.r.t to '+var)
        plt.show()

def visualizePieChart(X,cols):
    """
        Pie chart.
    """
    for var in cols:
        ax = X[var].value_counts().plot(kind='pie')
        ax.set_xlabel(var)
        ax.set_ylabel('Frequency')
        ax.set_title('Frequency Distribution for '+var)
        plt.show()