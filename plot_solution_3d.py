import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# from numpy import pi

# ## The function that we are trying to plot takes input (x, y, z) on the surface of interest

# def f(x, y, z=None):
# 	# if z is not None:
# 	# 	assert np.isclose(x**2 + y**2 + z**2, 1.0) ## check that this is on the unit sphere surface
# 	return y

# # domains
# # test data.
# data = np.genfromtxt("test.dat", delimiter=' ')
# # x = data[:, 0]
# # y = data[:, 1]
# # z = data[:, 2]

# x = np.linspace(-1, 1, 100)
# y = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.sqrt(1 - X**2 - Y**2)


# print(x, y, z)

# pred = data[:,3]
# pred = pred.reshape((pred.shape[0],1))
# # convert to 2d matrices
# # Z = np.outer(z.T, z)  
# X, Y = np.meshgrid(x, y) 
# # Z, _= np.meshgrid(z,x)
# Pred, _ = np.meshgrid(pred,x)

# print(X.shape)
# print(Y.shape)
# print(Z.shape)
# print(pred.shape)
# fourth dimension - colormap

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# domains
# test data.
data = np.genfromtxt("test.dat", delimiter=' ')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
true = data[:,3]
pred = data[:,4]
true = pred.reshape((true.shape[0],1))
pred = pred.reshape((pred.shape[0],1))
# convert to 2d matrices
X, Y, Z = np.meshgrid(x, y, z) 
Pred = np.outer(pred.T, pred)
fig = plt.figure()

# # print(X.shape)
# # print(Y.shape)
# # print(Z.shape)
# # print(pred.shape)
# # fourth dimention - colormap

# create colormap according to x-value (can use any 50x50 array)
color = f(X, Y, Z) # change to desired fourth dimension
minn, maxx = color.min(), color.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color)

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, shade=False)

ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# fig.canvas.show()
plt.show()

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
# import numpy as np

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


# pred = data[:, 3]

# # # polar coordinates to cartesian coordinates: 
# # X = r * np.sin(phi) * np.cos(theta)
# # Y = r * np.sin(phi) * np.sin(theta)
# # Z = r * np.cos(phi)

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, facecolors=cm.jet(pred),
#                        linewidth=0, antialiased=False)

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# 
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # test data.
# data = np.genfromtxt("test.dat", delimiter=' ')
# X = data[:, 0]
# Y = data[:, 1]
# Z = data[:, 2]
# pred = data[:,3]

# surf = ax.plot_trisurf(
#     X, Y, Z,
#     facecolors=cm.jet(pred),
#     linewidth=0, antialiased=False, shade=False)
# plt.show()