import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)

a=[]
b = [5,6]
c = [3,4]
f = [7,3]
a.append(c)
a.append(b)
a.append(f)

plt.plot(*zip(*a), marker='+', color='r', ls='')
plt.plot(*zip(*a),color = 'r')

plt.subplot(212)
plt.plot(*zip(*a), marker='+', color='b', ls='')
plt.plot(*zip(*a),color = 'b')

plt.show()
