"""Backend supported: tensorflow.compat.v1"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from deepxde.backend import tf

import deepxde as dde


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 1


def boundary(_, on_boundary):
    return on_boundary

## 2D L-shape Poisson Equation
#geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])

## S^1 Sphere Poisson Equation
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
bc_rad = dde.DirichletBC(
    geom,
    lambda x: np.cos(x[:, 1:2]),
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
)
data = dde.data.PDE(
    geom, pde, bc_rad, num_domain=1200, num_boundary=120, num_test=1500)

net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")

# Backend tensorflow.compat.v1 or tensorflow
def feature_transform(x):
    return tf.concat(
        [x[:, 0:1] * tf.sin(x[:, 1:2]), x[:, 0:1] * tf.cos(x[:, 1:2])], axis=1
    )
# Backend pytorch
# def feature_transform(x):
#     return torch.cat(
#         [x[:, 0:1] * torch.sin(x[:, 1:2]), x[:, 0:1] * torch.cos(x[:, 1:2])], dim=1
#     )

net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
model.compile("L-BFGS-B")
model.train(epochs=50000)
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

