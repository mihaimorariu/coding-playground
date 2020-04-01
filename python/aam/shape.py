import numpy as np
import pylab
import os

from matplotlib.mlab import PCA
from procrustes import Procrustes

class Shapes:
    def __init__(self):
        self.shapes = []
        self.paths = []

    def load(self, trainFiles):
        for trainFile in trainFiles:
            inputFile = open(trainFile, 'r')
            inputShape = []

            for i in range(0, 3):
                inputFile.readline()

            for i in range(4, 24):
                line = inputFile.readline()
                inputShape.extend(line.split())

            inputShape = [float(x) for x in inputShape]

            self.shapes.append(inputShape)
            self.paths.append(inputFile)
            inputFile.close()

        self.shapes = Shapes.convertToMat(self.shapes)
        print "Loaded", len(self.shapes), "training shapes."

    @staticmethod
    def mean(shapes):
        if len(shapes) == 0:
            pass

        mean = shapes[0]

        for i in range(1, len(shapes)):
            mean = mean + shapes[i]
        mean /= len(shapes)

        return mean

    def show(self, shapes):
        if len(shapes) == 0:
            pass

        for shape in shapes:
            pylab.plot(shape[:, 0], shape[:, 1], 'bo', markersize = 1)

        pylab.plot(self.mean[:, 0], self.mean[:, 1], 'ro', markersize = 4)
        pylab.gca().invert_yaxis()
        pylab.show()

    def align(self):
        procrustes = Procrustes(self.shapes)
        return procrustes.align(self.shapes)

    def pca(self, shapes):
        shapes_vec = Shape.to_array(shapes)
        return PCA(np.array(shapes_vec))

    @staticmethod
    def to_array(shapes):
        shapes_array = []

        for shape in shapes:
            shape_array = []
            for row in shape:
                shape_array.extend(row)
            shapes_array.append(shape_array)

        return shapes_array

    @staticmethod
    def convertToMat(shapes):
        if type(shapes).__name__ == 'ndarray':
            shapes_mat = np.array([[shapes[2*i], shapes[2*i + 1]] for i in range(0, len(shapes) / 2)])
        else:
            shapes_mat = [np.array([[x[2*i], x[2*i + 1]] for i in range(0, len(x) / 2)]) for x in shapes]

        return shapes_mat
