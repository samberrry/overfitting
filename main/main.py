import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_O_CLASS= 300

cov = [[3, 0], [0, 3]]

mean1 = [10, 4]

mean2 = [5, 15]

mean3 = [15, 15]

x1, y1 = np.random.multivariate_normal(mean1, cov, NUMBER_OF_O_CLASS).T
x2, y2 = np.random.multivariate_normal(mean2, cov, NUMBER_OF_O_CLASS).T
x3, y3 = np.random.multivariate_normal(mean3, cov, NUMBER_OF_O_CLASS).T

x = np.concatenate((x1,x2,x3),axis= 0)
y = np.concatenate((y1,y2,y3),axis= 0)

print(x2[0])
print(y2[0])
print(x[3])
print(y[3])

plt.plot(x, y, 'o')

samplx = np.random.uniform(low=0, high=20, size=(1800,))
samply = np.random.uniform(low=0, high=20, size=(1800,))

plt.plot(samplx, samply , "+")

plt.axis('equal')

plt.show()