import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)

def my_forward(x):
	h = l1(x)
	return l2(h)

# ----------------------------------------------

class MyProc(object):
	def __init__(self):
		self.l1 = L.Linear(4, 3)
		self.l2 = L.Linear(3, 2)

	def forward(self, x):
		h = self.l1(x)
		return self.l2(h)

# ----------------------------------------------

class MyChain(Chain):
	def __init__(self):
		super(MyChain, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(4, 3)
			self.l2 = L.Linear(3, 2)

	def __call__(self, x):
		h = self.l1(x)
		return self.l2(h)

# ----------------------------------------------

class MyChain2(ChainList):
	def __init__(self):
		super(MyChain2, self).__init__(
			L.Linear(4, 3),
			L.Linear(3, 2),
		)

	def __call__(self, x):
		h = self[0](x)
		return self[1](h)
