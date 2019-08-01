from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

def drow_pca(data, lables, name=''):
    pca_transformer = PCA(n_components=2).fit(data)
    pca = pca_transformer.transform(data)
    plt.scatter(pca[:, 0], pca[:, 1], 
                c=lables, s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(name+' PCA');