import cv2
import pylab
import os

from shape import Shape
from texture import Texture

def main():
    train_directory = '/home/mihai/Proiecte/AAM/Dataset/'
    shape_train_files = []
    texture_train_files = []

    for file_name in os.listdir(train_directory):
            if file_name.endswith(".pts"): 
                file_name = os.path.splitext(file_name)[0]
                file_path = os.path.join(train_directory, file_name)
                shape_train_files.append(file_path + '.pts')
                texture_train_files.append(file_path + '.pgm')

    train_shapes = Shape()
    train_shapes.load(shape_train_files)

    aligned_shapes = train_shapes.align()
    mean_shape = Shape.mean(aligned_shapes)
    pca = train_shapes.pca(train_shapes.shapes)

    train_textures = Texture()
    train_textures.load(texture_train_files, train_shapes.shapes, mean_shape)

#    pca = train_textures.pca(train_textures.textures)
#    train_textures.showDelaunay(train_shapes)

#    train_shapes.show()

#    texture = Texture(train_shapes.mean)
#    texture.showDelaunay(train_shapes.mean)    

if __name__ == '__main__':
    main()
