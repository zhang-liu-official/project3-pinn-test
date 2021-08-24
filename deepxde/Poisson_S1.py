"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from deepxde.backend import tf

def pde(x, y):
    dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
    # return x[:, 0:1] * dy_r + x[:, 0:1] ** 2 * dy_rr + dy_thetatheta
    lhs = x[:, 0:1] * dy_r + x[:, 0:1] * dy_rr + dy_thetatheta    
    # rhs = tf.sin( x[:, 0:1])
    rhs = 1.0
    return lhs - rhs 

def boundary(_, on_boundary):
    return on_boundary

# Backend tensorflow.compat.v1 or tensorflow
def feature_transform(x):
    return tf.concat(
        [x[:, 0:1] * tf.sin(x[:, 1:2]), x[:, 0:1] * tf.cos(x[:, 1:2])], axis=1
    )

def func(x,y):
    return (1-x**2-y**2)/4
# def solution(x):
#     r, theta = x[:, 0:1], x[:, 1:]
#     return r * np.cos(theta)
def main():
    # geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
    # unit sphere centered at (0,0,0) (radius = 1)
    # geom = dde.geometry.Sphere([0, 0, 0], 1)
    geom = dde.geometry.geometry_2d.Disk([0,0], radius = 1)

    bc = dde.ZeroLossBC(
    geom,
    func,
    boundary,
)

    data = dde.data.PDE(
        geom, pde, bc, num_domain=2500, num_boundary=0, num_test=2500, solution=func
        )

    net = dde.maps.FNN([2] + [20] * 4 + [1], "tanh", "Glorot uniform")

    net.apply_feature_transform(feature_transform)

    # Use [r*sin(theta), r*cos(theta)] as features,
    # so that the network is automatically periodic along the theta coordinate.
    # Backend tensorflow.compat.v1 or tensorflow

    # Backend pytorch
    # def feature_transform(x):
    #     return torch.cat(
    #         [x[:, 0:1] * torch.sin(x[:, 1:2]), x[:, 0:1] * torch.cos(x[:, 1:2])], dim=1
    #     )

    # net.apply_feature_transform(feature_transform)

    model = dde.Model(data, net)
    
    model.compile("adam", lr=0.001)
    model.train(epochs=15000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    X = geom.uniform_points(2500)
    y_true = func(X)
    # y_pred is PDE residual
    y_pred = model.predict(X, operator = pde)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

if __name__ == "__main__":
    main()