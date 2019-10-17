import numpy as np

"""
This file implements a common first-order update rule stochastic gradient descent used
for training neural networks. The update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""

def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

def rmsprop(w, dw, config=None):
    """
    RMSprop

    Not seems to be working
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("beta", 0.9)
    config.setdefault("grad_squared", 0)

    config["grad_squared"] = config["beta"] * config["grad_squared"] + (1 - config["beta"]) * dw * dw
    w -= (config["learning_rate"] / np.sqrt(config["grad_squared"]) + 1e-8) * dw
    return w, config


def adam(w, dw, config=None, eps=1e-8):
    """
    Adam
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("beta_1", 0.9)
    config.setdefault("beta_2", 0.999)

    config.setdefault("t", 0)
    config.setdefault("m", 0)
    config.setdefault("v", 0)

    config["t"] += 1
    config["m"] = config["beta_1"] * config["m"] + (1 - config["beta_1"]) * dw
    config["v"] = config["beta_2"] * config["v"] + (1 - config["beta_2"]) * dw ** 2
    config["m"] = config["m"] / (1 - config["beta_1"] ** config["t"])
    config["v"] = config["v"] / (1 - config["beta_2"] ** config["t"])
    w -= config["learning_rate"] * config["m"] / (np.sqrt(config["v"] + eps))

    return w, config
