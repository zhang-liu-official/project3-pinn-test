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
    return x[:, 0:1] * dy_r + x[:, 0:1] ** 2 * dy_rr + dy_thetatheta


def boundary(_, on_boundary):
    return on_boundary

# Backend tensorflow.compat.v1 or tensorflow
def feature_transform(x):
    return tf.concat(
        [x[:, 0:1] * tf.sin(x[:, 1:2]), x[:, 0:1] * tf.cos(x[:, 1:2])], axis=1
    )

geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
# unit sphere centered at (0,0,0) (radius = 1)
# geom = dde.geometry.Sphere([0, 0, 0], 1)

bc = dde.ZeroLossBC(geom, lambda _, on_boundary: on_boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=0, num_test=1500)

net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")

net.apply_feature_transform(feature_transform)

# Use [r*sin(theta), r*cos(theta)] as features,
# so that the network is automatically periodic along the theta coordinate.
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

model.compile("adam", lr=0.001)
model.train(epochs=50000)
model.compile("L-BFGS-B")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X = geom.uniform_points(50000)
y_true = func(X)
# y_pred is PDE residual
y_pred = model.predict(X, operator = pde)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
