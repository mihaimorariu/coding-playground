import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

x_data = np.array([5], dtype=np.float32)
x      = Variable(x_data)
y      = x**2 - 2*x + 1

#print(y.data)
y.backward()
#print(x.grad)

# ----------------------------------------------

z = 2*x
y = x**2 - z + 1
y.backward(retain_grad=True)
#print(z.grad)

y.backward()
#print(z.grad)

# ----------------------------------------------

x      = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y      = x**2 - 2*x + 1
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
#print(x.grad)

# ----------------------------------------------

f = L.Linear(3, 2)
#print(f.W.data)
#print(f.b.data)
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
f.cleargrads()
#print(y.data)
y.grad = np.ones((2, 2), dtype=np.float32)
y.backward()
print(f.W.grad)
print(f.b.grad)
