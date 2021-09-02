"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch

Documentation: https://deepxde.readthedocs.io/en/latest/demos/poisson.1d.dirichlet.html
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
# Import torch if using backend pytorch
# import torch


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # Use tf.sin for backend tensorflow.compat.v1 or tensorflow
    return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)
    # Use torch.sin for backend pytorch
    # return -dy_xx - np.pi ** 2 * torch.sin(np.pi * x)


# def boundary(x, on_boundary):
#     return on_boundary


def solution(x):
    return np.sin(np.pi * x)
def boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[0], -1) or np.isclose(x[0], 1))
def func(x):
    return 0
geom = dde.geometry.Interval(-1, 1)
bc = dde.DirichletBC(geom, func, boundary)
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=solution, num_test=100)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(epochs=10000)
# Optional: Save the model during training.
# checkpointer = dde.callbacks.ModelCheckpoint(
#     "model/model.ckpt", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = dde.callbacks.MovieDumper(
#     "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
# )
# losshistory, train_state = model.train(epochs=10000, callbacks=[checkpointer, movie])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Optional: Restore the saved model with the smallest training loss
# model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
# Plot PDE residual
x = geom.uniform_points(1000, True)
y = model.predict(x, operator=pde)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.show()
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
