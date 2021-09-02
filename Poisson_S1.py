"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import xde as xde
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from deepxde.backend import tf

## useful reference: https://en.wikipedia.org/wiki/Laplace_operator#Coordinate_expressions
## Laplacian-beltrami operator in spherical coordinates for 2-sphere:
##                   https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator#Examples

# note thta u (the solution) is the y here!
def pde(x, y):
    ## Poisson vs Laplacian: only diff is the rhs is f(x) vs 0
    
    # r = x[:, 0:1] is the radial distance and theta = x[:, 1:] is the angle

    theta = x[:]
    # dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    # dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # lhs = dy_xx + dy_yy

    # dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    # dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    # dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
    # lhs = 1/ r * dy_r +   dy_rr + 1/ (r ** 2) * dy_thetatheta    
    ## if laplace equation: 
    # rhs = 0

    dy_thetatheta = dde.grad.hessian(y, x, i=0, j=0)
    lhs = dy_thetatheta
    rhs = tf.sin(theta)
    return lhs - rhs 

def boundary(x, _):
    ## (Note that because of rounding-off errors, it is often wise to use np.isclose to test whether two floating point values are equivalent.)
    return np.isclose(x[0], 0) or np.isclose(x[0], 2*np.pi)

def func(x):
    return 0

# def boundary(x, on_boundary):
#     return on_boundary

# def func(x):
#     theta = x[:]
#     return -np.sin(theta)

def solution(x):
    theta = x[:]
    ## if laplacian, the solution is:
    # return r  * np.cos(theta) 
    return -np.sin(theta)

# Use [r*sin(theta), r*cos(theta)] as features,
# so that the network is automatically periodic along the theta coordinate.
# Backend tensorflow.compat.v1 or tensorflow
# def feature_transform(x):
#     return tf.concat(
#         [tf.sin(x[:]), tf.cos(x[:])], axis=1 ## since r = 1 
#     )

def main():
    # geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
    # unit sphere centered at (0,0,0) (radius = 1)
    # geom = dde.geometry.geometry_nd.Hypersphere([0,0], radius = 1)
    geom = dde.geometry.geometry_1d.Interval(0, 2 * np.pi)

    ## BC: u(0) = u(2 * pi) = 0
    bc = dde.DirichletBC(
        geom,
        func, 
        boundary,
    )

    # bc = xde.ZeroLossBC(geom, func, boundary)
    data = dde.data.PDE(
        geom, pde, bc, num_domain=100, num_boundary=2, num_test = 80, solution = solution)
    ## original NN parameters
    net = dde.maps.FNN([1] + [500]  + [1], "tanh", "Glorot uniform")

    ## over-parameterized
    # net = dde.maps.FNN([2] + [1200]*2  + [1], "tanh", "Glorot uniform")

    # net.apply_feature_transform(feature_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
    losshistory, train_state = model.train(epochs=15000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    ## uniform_points not implemented for hypersphere. test data used random_points instead, following distribution defined here: https://mathworld.wolfram.com/DiskPointPicking.html
    X = geom.uniform_points(1000)
    # X = feature_transform(X)
    y_true = solution(X)
    # y_pred is PDE residual
    y_pred = model.predict(X, operator = pde)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    y_true = y_true.reshape((y_true.shape[0],1))
    y_pred = y_pred.reshape((y_pred.shape[0],1))
    np.savetxt("test.dat", np.hstack((X,y_true, y_pred)))

if __name__ == "__main__":
    main()