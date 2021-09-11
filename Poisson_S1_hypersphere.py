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

    # (X1, X2) = (x,y) = (cos(theta), sin(theta))
    X1, X2= x[:, 0], x[:,1]
    X1 = tf.reshape(X1, (X1.shape[0],1))
    X2 = tf.reshape(X2, (X2.shape[0],1))
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    lhs = dy_xx + dy_yy
    # sin(theta)
    rhs = X2
    return lhs - rhs 

def boundary(x, on_boundary):
    ## (Note that because of rounding-off errors, it is often wise to use np.isclose to test whether two floating point values are equivalent.)
    return on_boundary 

def solution(x):
    # (X1, X2) = (x,y) = (cos(theta), sin(theta))
    X1, X2= x[:, 0], x[:,1]
    X1 = tf.reshape(X1, (X1.shape[0],1))
    X2 = tf.reshape(X2, (X2.shape[0],1))
    ## if laplacian, the solution is:
    # return r  * np.cos(theta) 

    ##-np.sin(theta)
    return -X2

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
    geom = xde.geometry.geometry_nd.Hypersphere([0,0], radius = 1)

    ## BC: u(0) = u(2 * pi) 
    bc = xde.ZeroLossBC(
        geom,
        lambda x: x,
        boundary,
    )

    # bc = xde.ZeroLossBC(geom, func, boundary)
    data = dde.data.PDE(
        geom, pde,  [bc], num_domain=400, num_boundary=0, num_test = 80, solution = solution)
    ## original NN parameters
    net = dde.maps.FNN([2] + [500]  + [1], "tanh", "Glorot uniform")

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