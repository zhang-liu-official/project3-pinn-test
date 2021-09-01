import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = np.genfromtxt("test.dat", delimiter=' ')
x = data[:, 0]
y = data[:, 1]
z1 = data[:, 2]
z2 = data[:, 3]

fig, ax = plt.subplots(nrows=1)
ax.tricontour(x, y, z1, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(x, y, z1, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(x, y, 'k.', ms=3)
plt.show()

fig, ax = plt.subplots(nrows=1)
ax.tricontour(x, y, z2, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(x, y, z2, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(x, y, 'k.', ms=3)
plt.show()