# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # domains
# x = np.linspace(0, 1,50) # [0.1, 5]
# y = np.linspace(6,9,50)             # [6, 9]
# z = np.linspace(-1,1,50)            # [-1, 1]

# # convert to 2d matrices
# Z = np.outer(z.T, z)        # 50x50
# X, Y = np.meshgrid(x, y)    # 50x50

# print(x.shape, X.shape)
# print(y.shape, Y.shape)
# print(z.shape, Z.shape)

# # fourth dimention - colormap
# # create colormap according to x-value (can use any 50x50 array)
# color_dimension = X # change to desired fourth dimension
# minn, maxx = color_dimension.min(), color_dimension.max()
# norm = matplotlib.colors.Normalize(minn, maxx)
# m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
# m.set_array([])
# fcolors = m.to_rgba(color_dimension)

# # plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

# ### Second example

# # import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
# # import numpy as np

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# # Z = np.sin(R)

# print(x.shape, X.shape)
# print(y.shape, Y.shape)
# print(z.shape, Z.shape)

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
# import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

print(x.shape, X.shape)
print(y.shape, Y.shape)
print(z.shape, Z.shape)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()