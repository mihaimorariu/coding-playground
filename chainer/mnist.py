#!/usr/bin/env python

from __future__ import print_function

import argparse
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training
from chainer.training import extensions


class MLP(chainer.Chain):
	def __init__(self, n_units, n_out):
		super(MLP, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, n_units)
			self.l2 = L.Linear(None, n_units)
			self.l3 = L.Linear(None, n_out)

	def __call__(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		return self.l3(h2)


def main():
	parser = argparse.ArgumentParser(description='Chainer example: MNIST')
	parser.add_argument('--batchsize', '-b', type=int, default=100,      help='Number of images in each mini-batch')
	parser.add_argument('--epoch',     '-e', type=int, default=20,       help='Number of sweeps over the dataset to train')
	parser.add_argument('--frequency', '-f', type=int, default=-1,       help='Frequency of taking a snapshot')
	parser.add_argument('--gpu',       '-g', type=int, default=-1,       help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out',       '-o',           default='result', help='Directory to output the result')
	parser.add_argument('--resume',    '-r',           default='',       help='Resume the training from snapshot')
	parser.add_argument('--unit',      '-u', type=int, default=1000,     help='Number of units')
	parser.add_argument('--noplot', dest='plot', action='store_false',   help='Disable PlotReport extension')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# unit: {}'.format(args.unit))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')

	model = L.Classifier(MLP(args.unit, 10))
	if args.gpu >= 0:
		chainer.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	train, test = chainer.datasets.get_mnist()
	train_iter  = chainer.iterators.SerialIterator(train, args.batchsize, shuffle=True)
	test_iter   = chainer.iterators.SerialIterator(test,  args.batchsize, shuffle=False, repeat=False)

	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
	trainer.extend(extensions.dump_graph('main/loss'))

	frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
	trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
	trainer.extend(extensions.LogReport())

	if args.plot and extensions.PlotReport.available():
		trainer.extend(extensions.PlotReport(['main/loss',     'validation/main/loss'],     'epoch', file_name='loss.png'))
		trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

	trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
	trainer.extend(extensions.ProgressBar())

	if args.resume:
		chainer.serializers.load_npz(args.resume, trainer)

	trainer.run()


if __name__ == '__main__':
	main()
