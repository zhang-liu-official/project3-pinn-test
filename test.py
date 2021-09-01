"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import xde as xde
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from deepxde.backend import tf

## useful reference: https://en.wikipedia.org/wiki/Laplace_operator#Coordinate_expressions

# note thta u (the solution) is the y here!
def pde(x, y):
    ## adapted from Laplacian on disk example
    ## Poisson vs Laplacian: only diff is the rhs f(x) vs 0
    
    # r = x[:, 0:1] is the radial distance and theta = x[:, 1:] is the angle

    r, theta = x[:, 0:1], x[:, 1:]
    dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
    lhs = r * dy_r +  (r ** 2) * dy_rr + dy_thetatheta    
    ## if laplace equation: 
    rhs = 0
    # rhs = (r **2) * np.sin(theta)
    return lhs - rhs 

def boundary(_, on_boundary):
    return on_boundary

# # Backend tensorflow.compat.v1 or tensorflow
def feature_transform(x):
    return np.concatenate(
        [x[:, 0:1] * np.sin(x[:, 1:2]), x[:, 0:1] * np.cos(x[:, 1:2])], axis=1
    )

def func(x):
    r, theta = x[:, 0:1], x[:, 1:]
    ## if laplacian, the solution is r * np.cos(theta)
    # return r * np.cos(theta)
    # return 1/np.sqrt(2) * r * np.sin(theta)
    # return (r **2) *np.cos(theta) * np.sin(theta) + (r **2) *np.sin(theta) * np.sin(theta) 
    return 1/(2*np.pi) * r * (np.cos(theta) * np.cos(theta) + np.sin(theta) * np.sin(theta))


# geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
# unit sphere centered at (0,0,0) (radius = 1)
# geom = dde.geometry.Sphere([0, 0, 0], 1)
geom = dde.geometry.geometry_2d.Disk([0,0], radius = 1)

bc = xde.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(
    geom, pde, bc, num_domain=1500, num_boundary=0, num_test = 1000)
## original NN parameters
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

## over-parameterized
# net = dde.maps.FNN([2] + [1500] + [1], "tanh", "Glorot uniform")

# Use [r*sin(theta), r*cos(theta)] as features,
# so that the network is automatically periodic along the theta coordinate.
# Backend tensorflow.compat.v1 or tensorflow

net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)
# model.compile("adam", lr=0.001)

# losshistory, train_state = model.train(epochs=10000)
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

## test data is random_points instead, following distribution defined here: https://mathworld.wolfram.com/DiskPointPicking.html
X = geom.uniform_points(1000)
X = feature_transform(X)
y_true = func(X)
# y_pred is PDE residual
# y_pred = model.predict(X, operator = pde)
# print("L2 relative error:", dde.metrics.l2_relative_error(y_pred))
# y_pred = y_pred.reshape((y_pred.shape[0],1))
np.savetxt("test.dat", np.hstack((X, y_true)))
