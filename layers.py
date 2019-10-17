from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass of affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) with mini-batch of N samples.
    Each sample x[i] has shape (d_1, ..., d_k). We will reshape each input into
    shape D = d_1 * ... * d_k, and then transform in to shape M.

    Inputs:
    -x: A numpy array containing input data, with shape (N, d_1, ..., d_k)
    -w: A numpy array of weights, shape (D, M)
    -b: A numpy array of biases, shape (M, )

    Returns a tuple of:
    -out: A numpy array of output data, shape (N, M)
    -cache: (x, w, b)
    """
    N = x.shape[0]
    D = w.shape[0]
    M = w.shape[1]
    cache = (x, w, b)

    x = np.reshape(x, (N, D))
    out = x @ w + b

    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass of affine (fully-connected) layer.

    Inputs:
    - dout: Upstream derivative, shape (N, M)
    - cache: Cache from forward pass. Tuple of (x, w, b)
     - x: Input data, shape (N, d_1, ..., d_k)
     - w: Weights, shape (D, M)
     - b: Biases, shape (M, )

    Returns a tuple of:
    - dx: Gradient of dout respect to x, shape (N, d_1, ..., d_k)
    - dw: Gradient of dout respect to w, shape (N, M)
    - db: Gradient of dout respect to b, shape (M, )
    """
    x, w, b = cache
    N = x.shape[0]
    D = w.shape[0]
    M = w.shape[1]

    dx = dout @ w.T
    dw = x.T @ dout
    db = np.sum(dout, axis=0)
    return dx, dw, db

def conv_forward(x, w, b, stride=1, padding=True):
    """
    Computes the forward pass for convolutional layer.

    Inputs:
    - x: A numpy array of input data of shape (N, d_1, d_2)
    - w: A numpy array of weights, shape (w, h, M)
    - b: A vector of bias terms, shape (M_1, M_2)
    - stride: Scalar for stride step size

    Returns a tuple of:
    - out: The output data of shape (N, M_1, M_2)
    - cache: (x, w, b)
    """
    pass

def conv_backward():
    pass

def relu_forward(x):
    """
    Computes the forward pass for ReLu (rectified linear) layer.

    Input:
    - x: A numpy array of any shape

    Returns a tuple of:
    - out: Output array of shape same as x
    - cache: x
    """
    out = np.maximum(x, 0.0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for ReLu layer.

    Input:
    - dout: Upstream derivative of any shape
    - cache: Input x, same shape as dout

    Returns:
    - dx: Gradient of dout respect to x, shape same as dout
    """
    is_non0 = cache > 0
    dx = dout * is_non0
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient at softmax layer (classification layer).

    Inputs:
    - x: A numpy array of data, shape (N, C) given C classes, where x[i, j] is the score of x[i] for jth class
    - y: Vector of the correct labels of shape (N, ) where y[i] contains the correct label for x[i]
      and 0 <= y[i] < C

    Returns:
    - loss: Scalar giving the loss
    - dx: Gradient of loss respect to x, same shape as x
    """
    loss = 0
    dx = []
    for i in range(x.shape[0]):
        m = np.max(x[i])
        t = np.exp(x[i]) / np.sum(np.exp(x[i]) - m, dtype=np.float64)
        loss += - (x[i][y[i]] - m) - np.log(np.sum(np.exp(x[i]) - m, dtype=np.float64))
        dx.append([-t[j] - 1 if j == y[i] else -t[j] for j in range(len(x[i]))])
    dx = np.asarray(dx)

    return loss, dx
