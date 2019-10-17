from gradient_check import eval_numerical_gradient_array
import numpy as np
from layers import *
N = 2
D = 3
M = 4
x = np.random.normal(size=(N, D))
w = np.random.normal(size=(D, M))
b = np.random.normal(size=(M, ))
# dout = np.random.normal(size=(N, M))
dout = np.random.normal(size=(N, D))

# out, cache = affine_forward(x, w, b)
# f=lambda x: affine_forward(x, w, b)[0]
# grad = affine_backward(dout, cache)[0]
# ngrad = eval_numerical_gradient_array(f, x, dout)
#
# print(grad - ngrad)

out, cache = relu_forward(x)
f=lambda x: relu_forward(x)[0]
grad = relu_backward(dout, cache)
ngrad = eval_numerical_gradient_array(f, x, dout)
print(grad - ngrad)
