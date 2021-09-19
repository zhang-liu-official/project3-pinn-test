"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
from deepxde.geometry.csg import CSGDifference
from numpy.core.numeric import isclose
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

    # (X1, X2, X3) = (x,y,z) = (sin(phi)cos(theta), sin(phi)sin(theta), cos(phi))
    theta, phi = x[:,0:1], x[:,1:]
    theta = tf.reshape(theta, [-1,1])
    phi = tf.reshape(phi, [-1,1])
    dy_thetatheta = dde.grad.hessian(y, x, i=0, j=0)
    dy_phi = dde.grad.jacobian(y, x,i=0,j=1)
    dy_phiphi = dde.grad.hessian(y,x, i=1, j=1)
    
    lhs = dy_phiphi * tf.sin(phi) ** 2 + tf.cos(phi) * tf.sin(phi) * dy_phi +  dy_thetatheta
    rhs = (3 * tf.cos(theta) ** 2 - 1) * tf.sin(phi) ** 2
    return lhs - rhs         

def solution(x):
    # print("******************************")
    # print(x)
    theta, phi = x[:,0:1], x[:,1:]
    ans = 3 * np.cos(theta) ** 2 - 1
    ans = ans.reshape((ans.shape[0], 1))

    ans = np.diag(np.full(ans.shape[0],-1/6)) @ ans

    ### remember to use diff scaling factors when the rhs is linear combination of the spherical harmonics 
    ### remember to do spherical harmonics
    return ans

def feature_transform(x):
    theta, phi = x[:,0:1], x[:,1:]
    theta = tf.reshape(theta, [-1,1])
    phi = tf.reshape(phi, [-1,1])

    return tf.concat(
        [tf.sin(phi) * tf.sin(theta),tf.sin(phi) * tf.cos(theta),   tf.cos(phi) ], axis=1 ## since r = 1 
    )
def main():

    ## create our own training points:
    # dde.data.PDE(..., anchors=X)
    # where X is a N by d matrix of total N points, with each row is one point of d-dim.
    # https://github.com/lululxvi/deepxde/issues/32
    ## Uniform density
    # Dataset size n
    # N = 5
    # n = N**2

    # # Spherical Coordinates: theta, phi
    # dtheta = 2 * np.pi/(2*N-1)
    # theta = np.repeat(np.arange(0, 2 * np.pi + dtheta/2, dtheta), N) 
    # # print(phi)
    # dcosphi = 2.0/(N-1)
    # cosphi = np.array([cost for i in range(2*N) for cost in np.arange(-1, 1 + dcosphi/2, dcosphi)])
    

    # theta in [0, 2pi] phi in [0, pi]
    geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2 * np.pi, np.pi])

    # geom1 = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2 * np.pi, 0.1])
    # geom2 = dde.geometry.Rectangle(xmin=[0, np.pi*0.9], xmax=[2 * np.pi, np.pi])
    # geom_excl = CSGDifference(geom, geom1)
    # geom_excl = CSGDifference(geom_excl, geom2)

    data = xde.data.PDE(
        geom, pde, [], num_domain=6000, num_boundary=2000, num_test = 10000, solution = solution)

    net = xde.maps.FNN([2] + [10000] + [1], "tanh", "Glorot uniform")
    net.apply_feature_transform(feature_transform)
    
    model = xde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
    losshistory, train_state = model.train(epochs=10000)
    xde.saveplot(losshistory, train_state, issave=True, isplot=True)

    ## uniform_points not implemented for hypersphere. test data used random_points instead, following distribution defined here: https://mathworld.wolfram.com/DiskPointPicking.html
    X = geom.uniform_points(100000)
    y_true = solution(X)
    # y_pred is PDE residual
    # x_0 = np.array([0.0, 0.0])
    # x_0 = x_0.reshape((1,2))
    # y_0 = solution(x_0)
    y_pred = model.predict(X, operator = pde)
    # y_pred = y_pred - y_0
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    y_true = y_true.reshape((y_true.shape[0],1))
    y_pred = y_pred.reshape((y_pred.shape[0],1))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
   

if __name__ == "__main__":
    main()