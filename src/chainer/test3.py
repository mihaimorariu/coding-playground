import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from test2 import MyChain

model     = MyChain()
optimizer = optimizers.SGD()

optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# ----------------------------------------------

x = np.random.uniform(-1, 1, (2, 4)).astype('f')
model.cleargrads()
loss = F.sum(model(chainer.Variable(x)))
loss.backward()
optimizer.update()

# ----------------------------------------------

def lossfun(arg1, arg2):
	return F.sum(model(arg1 - arg2))

arg1 = np.random.uniform(-1, 1, (2, 4)).astype('f')
arg2 = np.random.uniform(-1, 1, (2, 4)).astype('f')
optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))

# ----------------------------------------------

serializers.save_npz('my.model', model)
serializers.load_npz('my.model', model)
