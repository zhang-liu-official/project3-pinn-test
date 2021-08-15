from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import backend as bkd
from . import config
from .backend import tf
import numpy as np
import deepxde as dde

def mean_absolute_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    # Warning:
    # - Do not use ``tf.losses.mean_squared_error``, which casts `y_true` and `y_pred` to ``float32``.
    # - Do not use ``tf.keras.losses.MSE``, which computes the mean value over the last dimension.
    # - Do not use ``tf.keras.losses.MeanSquaredError()``, which casts loss to ``float32``
    #     when calling ``compute_weighted_loss()`` calling ``scale_losses_by_sample_weight()``,
    #     although it finally casts loss back to the original type.
    return bkd.reduce_mean(bkd.square(y_true - y_pred))


def softmax_cross_entropy(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def zero(*_):
    # TODO: pytorch
    return tf.constant(0, dtype=config.real(tf))

# def pde(x, y):
#         dy_xx = dde.grad.hessian(y, x, i=0, j=0)
#         dy_yy = dde.grad.hessian(y, x, i=1, j=1)
#         return -dy_xx - dy_yy - 1

# def boundary(_, on_boundary):
#         return on_boundary

def hs_norm(y_true, y_pred):
    # ## S^1 Sphere Poisson Equation
    # geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
    # bc_rad = dde.DirichletBC(
    #     geom,
    #     lambda x: np.cos(x[:, 1:2]),
    #     lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
    # )
    # data = dde.data.PDE(geom, pde, bc_rad, num_domain=1200, num_boundary=120)#, num_test=1500

    # net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
    u = y_true - y_pred
    n = 53
    if u.shape[0] is not None:
        n = u.shape[0]
    u = tf.cast(u, tf.float64)
    s = 0
    ## test here 
    # n = 1320
    dft_matrix = np.fft.fft(np.eye(n))
    inverse_dft_matrix = np.fft.ifft(np.eye(n))
    hs_weight_matrix = np.diag([(1 + i ** 2)**(s/2) for i in range(n)])
    result = np.matmul(inverse_dft_matrix, np.matmul(hs_weight_matrix, dft_matrix))
    hermitian_adjoint = np.matrix(result)
    result = np.matmul(hermitian_adjoint.getH(), result)
    P = np.array(result)
    
    ## # SciPy's L-BFGS-B Fortran implementation requires and returns float64
    P = tf.convert_to_tensor(P, dtype=tf.float64)
    hs_loss = 1.0/n * tf.matmul(tf.transpose(u), tf.matmul(P, u))
    return hs_loss

def get(identifier):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    loss_identifier = {
        "mean absolute error": mean_absolute_error,
        "MAE": mean_absolute_error,
        "mae": mean_absolute_error,
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "mean absolute percentage error": mean_absolute_percentage_error,
        "MAPE": mean_absolute_percentage_error,
        "mape": mean_absolute_percentage_error,
        "softmax cross entropy": softmax_cross_entropy,
        "zero": zero, 
        "Hs": hs_norm,
    }

    if isinstance(identifier, str):
        return loss_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
