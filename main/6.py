import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

# from pandas.tools.plotting import parallel_coordinates

linalg = np.linalg

X, y = make_blobs(n_samples=300, centers=3, n_features=5, random_state=0)

print(X)


plt.scatter(X[:,0], X[:,1], c=y)
plt.show()