import numpy as np

from layers import *

class FullyConnectedNet:
    """
    A fully-connected neural network with several options of activation layers
    and softmax loss that uses a modular layer design. We assume the input has
    dimension D, hidden layers with dimensions [H, ...] and perform classification
    over C classes.

    The architecure should be like affine - relu - affine - softmax for one hidden
    layer network, and affine -relu - affine - relu - affine - softmax for two
    hidden layer, etc.

    Note that this class does not have any gradient descent method; instead
    we will call a separate Solver object to perform optimization.

    The parameters of the model are stored in a dictionary self.params which maps
    parameter names to arrays
    """

    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=10, weight_scale=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.input_dim = input_dim
        self.hidden_num = len(hidden_dim)

        self.params["W1"] = np.random.normal(0, weight_scale, (self.input_dim, hidden_dim[0]))
        self.params["b1"] = np.zeros((hidden_dim[0], ))
        for n in np.arange(2, self.hidden_num + 1):
            self.params["W" + str(n)] = np.random.normal(0, weight_scale, (hidden_dim[n - 2], hidden_dim[n - 1]))
            self.params["b" + str(n)] = np.zeros((hidden_dim[n - 1], ))
        self.params["W" + str(n + 1)] = np.random.normal(0, weight_scale, (hidden_dim[n - 1], num_classes))
        self.params["b" + str(n + 1)] = np.zeros((num_classes, ))


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        cache, out = {}, X
        for n in range(1, self.hidden_num + 1):
            out, cache["fc" + str(n)] = affine_forward(out, self.params["W" + str(n)], self.params["b" + str(n)])
            out, cache["rl" + str(n)] = relu_forward(out)
        scores, cache["fc" + str(n + 1)] = affine_forward(out, self.params["W" + str(n + 1)], self.params["b" + str(n + 1)])

        #when y is none, it is running on test set
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dx = softmax_loss(scores, y)
        dx, grads["W" + str(n + 1)], grads["b" + str(n + 1)] = affine_backward(dx, cache["fc" + str(n + 1)])
        for n in reversed(range(1, self.hidden_num + 1)):
            dx = relu_backward(dx, cache["rl" + str(n)])
            dx, grads["W" + str(n)], grads["b" + str(n)] = affine_backward(dx, cache["fc" + str(n)])

        return loss, grads
