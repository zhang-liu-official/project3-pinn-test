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

    # (X1, X2, X3) = (x,y,z) = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    X1, X2, X3 = x[:, 0], x[:,1], x[:,2]
    theta = tf.math.atan(tf.math.divide(X2, X1))
    theta = tf.reshape(theta, [-1,1])
    # phi = tf.math.acos(X3)
    # phi = tf.reshape(phi, [-1,1])
    # phi =  tf.reshape(phi, [phi.shape[0],1])
    dy_xx = xde.grad.hessian(y, x, i=0, j=0)
    dy_yy = xde.grad.hessian(y, x, i=1, j=1)
    dy_zz = xde.grad.hessian(y, x, i=2, j=2)
    lhs = dy_xx + dy_yy + dy_zz

    
    # dy_thetatheta = dde.grad.hessian(y, theta, i=0, j=0)
    # dy_phi = dde.grad.jacobian(y, phi, i=0, j=0)
    # dy_phiphi = dde.grad.hessian(y,phi, i=0, j=0)
    # lhs = dy_phiphi + tf.cos(phi)/tf.sin(phi) * dy_phi + tf.sin(phi) **(-2) * dy_thetatheta
    rhs = 3 * tf.cos(theta) ** 2 - 1

    return lhs - rhs 

def boundary(x, on_boundary):
    return on_boundary

def main():
    # geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
    # unit sphere centered at (0,0,0) (radius = 1)
    # geom = dde.geometry.Sphere([0, 0, 0], 1)
    geom = xde.geometry.geometry_nd.Hypersphere([0,0,0], radius = 1)

    # bc = dde.DirichletBC(
    #     geom,
    #     lambda x: np.cos(x[:, 1:2]),
    #     lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
    # )

    bc = xde.ZeroLossBC(
        geom,
        boundary,
    )

    data = xde.data.PDE(
        geom, pde, bc, num_domain=500, num_boundary= 0, num_test = 1500)

    net = xde.maps.FNN([3] + [1000]  + [1], "tanh", "Glorot uniform")

    model = xde.Model(data, net)
    model.compile("adam", lr=0.001)
    
    losshistory, train_state = model.train(epochs=15000)
    xde.saveplot(losshistory, train_state, issave=True, isplot=True)

    ## uniform_points not implemented for hypersphere. test data used random_points instead, following distribution defined here: https://mathworld.wolfram.com/DiskPointPicking.html
    X = geom.uniform_points(1000)
    # y_true = solution(X)
    # y_pred is PDE residual
    y_pred = model.predict(X, operator = pde)
    # print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    # y_true = y_true.reshape((y_true.shape[0],1))
    y_pred = y_pred.reshape((y_pred.shape[0],1))
    np.savetxt("test.dat", np.hstack((X, y_pred)))

if __name__ == "__main__":
    main()