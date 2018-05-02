import matplotlib.pyplot as plt
import numpy as np



cov = [[1, 0], [0, 1]]

mean2 = [5, 15]

x, y = np.random.multivariate_normal(mean2, cov, 300).T


plt.plot(x, y, 'o')

plt.axis('equal')

plt.show()