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
    ## adapted from Laplacian on disk example
    ## Poisson vs Laplacian: only diff is the rhs f(x) vs 0
    
    # r = x[:, 0:1] is the radial distance and theta = x[:, 1:] is the angle


    ## need to double check whether x[:,2:] is actually phi
    r, theta, phi = x[:, 0:1], x[:, 1:2], x[:,2:]
    
    dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
    dy_phiphi = dde.grad.hessian(y, x, i=2, j=2)
    dy_phi = dde.grad.jacobian(y, x, i=2, j=2)
    ddy_phi = dde.grad.jacobian(tf.sin(phi) * dy_phi, x, i=2, j=2)
    lhs = tf.sin(phi) **(-1) * ddy_phi + tf.sin(phi) **(-2) * dy_thetatheta
    rhs = 3 * tf.cos(theta) ** 2 - 1
    return lhs - rhs 

def boundary(_, on_boundary):
    return on_boundary

# # Backend tensorflow.compat.v1 or tensorflow
# def feature_transform(x):
#     return tf.concat(
#         [x[:, 0:1] * tf.sin(x[:, 1:2]) * tf.cos(x[:, 2:]),  x[:, 0:1] * tf.sin(x[:, 1:2]) * tf.sin(x[:, 2:]), x[:, 0:1] * tf.cos(x[:, 1:2])], axis=1
#     )

def func(x):
    r, theta, phi = x[:, 0:1], x[:, 1:2], x[:,2:]
    ## if laplacian, the solution is:
    # return r  * np.cos(theta) 
    return - 1/2 * np.sin(theta)

def main():
    # geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
    # unit sphere centered at (0,0,0) (radius = 1)
    # geom = dde.geometry.Sphere([0, 0, 0], 1)
    geom = dde.geometry.geometry_nd.Hypersphere([0,0,0], radius = 1)

    # bc = dde.DirichletBC(
    #     geom,
    #     lambda x: np.cos(x[:, 1:2]),
    #     lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
    # )

    bc = xde.ZeroLossBC(
        geom,
        func,
        boundary,
    )

    data = dde.data.PDE(
        geom, pde, bc, num_domain=2500, num_boundary=0, num_test = 1500)
    ## original NN parameters
    net = dde.maps.FNN([3] + [50] * 4 + [1], "tanh", "Glorot normal")

    ## over-parameterized
    # net = dde.maps.FNN([2] + [1500] + [1], "tanh", "Glorot uniform")

    # Use [r*sin(theta), r*cos(theta)] as features,
    # so that the network is automatically periodic along the theta coordinate.
    # Backend tensorflow.compat.v1 or tensorflow

    # net.apply_feature_transform(feature_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    
    losshistory, train_state = model.train(epochs=20000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    ## test data is random_points instead, following distribution defined here: https://mathworld.wolfram.com/DiskPointPicking.html
    X = geom.uniform_points(1500)
    # y_true = func(X)
    # y_pred is PDE residual
    y_pred = model.predict(X, operator = pde)
    # print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    # np.savetxt("test.dat", np.hstack((X,y_true)))
    np.savetxt("test.dat", np.hstack((X,y_pred)))

if __name__ == "__main__":
    main()