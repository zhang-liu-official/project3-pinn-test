# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

# data = np.genfromtxt("test.dat", delimiter=' ')
# x = data[:, 0]
# y = data[:, 1]
# z = data[:, 2]
# pred = data[:, 3]

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
# cntr = ax.tricontourf(x, y, z, pred, levels=14, cmap="RdBu_r")
# fig.colorbar(cntr, ax=ax)
# ax.plot(x, y, z, 'k.', ms=3)
# plt.show()

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# test data.
data = np.genfromtxt("test.dat", delimiter=' ')
r = data[:, 0]
theta = data[:, 1]
phi = data[:, 2]
pred = data[:, 3]

# polar coordinates to cartesian coordinates: 
X = r * np.sin(phi) * np.cos(theta)
Y = r * np.sin(phi) * np.sin(theta)
Z = r * np.cos(phi)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, facecolors=cm.jet(pred),
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()