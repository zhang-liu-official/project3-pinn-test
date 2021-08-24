from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import backend as bkd
from . import config
from .backend import tf
import deepxde as dde
import numpy as np

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




def hs_norm(y_true, y_pred):
    # def pde(x, y):
    #     dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    #     dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    #     dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
    #     # return x[:, 0:1] * dy_r + x[:, 0:1] ** 2 * dy_rr + dy_thetatheta
    #     lhs = x[:, 0:1] * dy_r + x[:, 0:1] * dy_rr + dy_thetatheta    
    #     rhs = 1.0* tf.sin( x[:, 0:1])
    #     return lhs - rhs 

    # def boundary(_, on_boundary):
    #     return on_boundary

    # # Backend tensorflow.compat.v1 or tensorflow
    # def feature_transform(x):
    #     return tf.concat(
    #         [x[:, 0:1] * tf.sin(x[:, 1:2]), x[:, 0:1] * tf.cos(x[:, 1:2])], axis=1
    #     )
    #     ## S^1 Sphere Poisson Equation
    # geom = dde.geometry.geometry_2d.Disk([0,0], radius = 1)

    # bc = dde.ZeroLossBC(
    #     geom,
    #     lambda x: np.cos(x[:, 1:2]),
    #     lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
    # )

    # data = dde.data.PDE(
    #     geom, pde, bc, num_domain=2500, num_boundary=0, num_test=2500
    #     # solution=func
    #     )

    # net = dde.maps.FNN([2] + [20] * 4 + [1], "tanh", "Glorot normal")
    # net.apply_feature_transform(feature_transform)

    # # model = dde.Model(data, net)
    # # n = tf.shape(model.net.outputs)[0].value
   
    # # X_train = dde.data.DataSet().X_train
    # # print(X_train)
    # # variables_names = [v.name for v in tf.trainable_variables()]
    # # shapes = [v.get_shape().as_list() for v in tf.trainable_variables()]
    # # print(variables_names)
    # # print(shapes)
    
    # model = dde.Model(data, net)
    # # n = model.train_state.y_train.shape[0]
    # n = y_pred.shape
    u = y_true - y_pred
    u = tf.cast(u, tf.float64)
    # s = -1
    # n=2500
    # dft_matrix = np.fft.fft(np.eye(n))
    # inverse_dft_matrix = np.fft.ifft(np.eye(n))
    # hs_weight_matrix = np.diag([(1 + i ** 2)**(s/2) for i in range(n)])
    # result = np.matmul(inverse_dft_matrix, np.matmul(hs_weight_matrix, dft_matrix))
    # hermitian_adjoint = np.matrix(result)
    # result = np.matmul(hermitian_adjoint.getH(), result)
    # P = np.array(result)
    
    # ## # SciPy's L-BFGS-B Fortran implementation requires and returns float64
    # P = tf.convert_to_tensor(P, dtype=tf.float64)

    # hs_loss = 1.0/n * tf.matmul(tf.transpose(u), tf.matmul(P, u))
    return u


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
