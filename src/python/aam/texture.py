import cv2
import matplotlib.pyplot as pyplot
import numpy as np

from constants import *
from matplotlib.mlab import PCA
from scipy.spatial import Delaunay
from scipy.misc import imread

class Texture:
    def __init__(self):
        self.textures = []
        pass

    def create_face_mask(self, image, triangles, shape):
        rows, cols = image.shape
        binary_mask = np.zeros((rows, cols), np.uint8)

        for i in range(0, len(triangles.vertices)):
            vertex_a = (int(shape[triangles.vertices[i, 0], 0]), int(shape[triangles.vertices[i, 0], 1]))
            vertex_b = (int(shape[triangles.vertices[i, 1], 0]), int(shape[triangles.vertices[i, 1], 1]))
            vertex_c = (int(shape[triangles.vertices[i, 2], 0]), int(shape[triangles.vertices[i, 2], 1]))

            cv2.fillPoly(binary_mask, np.array([[vertex_a, vertex_b, vertex_c]]), white)
            face_mask = cv2.bitwise_and(image, binary_mask)

        return binary_mask, face_mask

    def create_feature_vector(self, image, shape, mean_shape, mean_triangles):
        H, mask = cv2.findHomography(shape, mean_shape)
        warped_image = cv2.warpPerspective(image, H, warped_image_size)
        binary_mask, face_mask = self.create_face_mask(warped_image, mean_triangles, mean_shape)

        cv2.imshow("Train image", face_mask)
        cv2.waitKey(0)
        pyplot.show()

        feature_vector = []
        for i in range(0, face_mask.shape[0]):
            for j in range(0, face_mask.shape[1]):
                if binary_mask[i][j] != 0:
                    feature_vector.append(face_mask[i][j])

        return feature_vector

    def normalize_feature_vector(self, vector):
        beta = np.sum(vector) / len(vector)
        norm_vector = vector - beta
        alpha = np.linalg.norm(norm_vector)
        norm_vector = norm_vector / alpha

        return norm_vector

    def load(self, train_files, train_shapes, mean_shape):
        if len(train_files) == 0 or len(train_shapes) == 0:
            pass

        image = imread(train_files[0])
        rows, cols = warped_image_size
        mean_shape = mean_shape * mean_shape_scale + np.tile([cols / 2, rows / 3], (len(mean_shape), 1))
        mean_triangles = Delaunay(mean_shape)

        for i in range(0, len(train_files)):
            image = imread(train_files[i])
            shape = train_shapes[i]

            feature_vector = self.create_feature_vector(image, shape, mean_shape, mean_triangles)
            norm_vector = self.normalize_feature_vector(feature_vector)
#            print len(norm_vector)

            self.textures.append(norm_vector)

        print "Loaded", len(self.textures), "training textures."

    def pca(self, textures):
        return PCA(np.array(textures))

#    def show_delaunay(self, shape):
#        pyplot.triplot(shape[:, 0], shape[:, 1], self.triangles.vertices.copy())
#        pyplot.plot(shape[:, 0], shape[:, 1], 'o')
#        pyplot.gca().invert_yaxis()
#        pyplot.show()
