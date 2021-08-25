  
import numpy as np
from matplotlib import pyplot

data = np.genfromtxt("test.dat", delimiter=' ')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

fig, ax = pyplot.subplots(nrows=1)
ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(x, y, 'k.', ms=3)
pyplot.show()