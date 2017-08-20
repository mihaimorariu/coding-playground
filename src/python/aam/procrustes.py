import numpy as np

from sys import exit

class Procrustes:
    def __init__(self, shapes):
        self.shapes = shapes

    def __align(self, shape1, shape2):
        norm_shape_1 = self.__normalize(shape1)
        norm_shape_2 = self.__normalize(shape2)

        U, S, V = np.linalg.svd(norm_shape_2.transpose().dot(norm_shape_1))
        R = U.dot(V);
        norm_shape_2 = norm_shape_2.dot(R)

        return norm_shape_2

    def __normalize(self, shape):
        centroid = np.array(np.mean(shape, axis = 0))
        normalized = shape - np.tile(centroid, (len(shape), 1))
        normalized = normalized / np.linalg.norm(normalized)  

        return normalized

    def align(self, shapes):
        print "Starting to align shapes..."

        aligned_shapes = list(shapes)
        last_mean_shape = self.__normalize(shapes[0])
        epsilon = 0.0000000000001
        k = 0

        while True:
            for i in range(len(shapes)):
                aligned_shapes[i] = self.__align(last_mean_shape, aligned_shapes[i])

            mean_shape = aligned_shapes[0]
            for i in range(1, len(shapes)):
                mean_shape = mean_shape + aligned_shapes[i]

            mean_shape = mean_shape / len(shapes)


            norm = np.linalg.norm(mean_shape - last_mean_shape)
            k = k + 1
            print "Iter:", k, "- Norm:", norm

            if norm < epsilon:
                break

            last_mean_shape = mean_shape


        print "Finished aligning shapes in ", k, " iterations."
        return aligned_shapes
