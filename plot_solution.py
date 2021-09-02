import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = np.genfromtxt("test.dat", delimiter=' ')
theta = data[:, 0]
x = np.cos(theta)
y = np.sin(theta)
true = data[:, 1]
pred = data[:, 2]

fig, ax = plt.subplots(nrows=1)
ax.tricontour(x, y, true, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(x, y, true, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(x, y, 'k.', ms=3)
plt.show()

fig, ax = plt.subplots(nrows=1)
ax.tricontour(x, y, pred, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(x, y, pred, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(x, y, 'k.', ms=3)
plt.show()