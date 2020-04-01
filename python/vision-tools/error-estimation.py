import yaml
import cv2
import math
import numpy as np
import sys, getopt

calib_file = None
img_points = None
z_depth = None

def get3DCoordinates(img_point, inv_cam_matrix):
  x, y = img_point
  direction = inv_cam_matrix * np.matrix([x, y, 1]).T
  alpha = z_depth / direction[2, 0]
  reproj = alpha * direction

  return (reproj[0, 0], reproj[1, 0], reproj[2, 0])

def printUsage():
  print(
    """error_estimation.py [calib_file] [x0] [y0] [x1] [y1] [depth]
    where the values represent:

    calib_file - Camera calibration file
     (x0, y0)  - Coordinates of an image point (ground truth)
     (x1, y1)  - Coordinates of an image point (detection)
       depth   - Depth at which the image points are back projected"""
  )

def parseArguments():
  global calib_file, img_points, z_depth

  if len(sys.argv) != 7:
    printUsage()
    exit(0)

  calib_file = sys.argv[1]
  img_points = []

  try:
    img_points.append((float(sys.argv[2]), float(sys.argv[3])))
    img_points.append((float(sys.argv[4]), float(sys.argv[5])))
    z_depth = float(sys.argv[6])
  except ValueError:
    printUsage()

def runEstimation():
  global img_points, z_depth

  with open(calib_file, "r") as f:
      data = yaml.load(f)

  cam_matrix = np.matrix(data["cam_matrix"])
  inv_cam_matrix = np.linalg.inv(cam_matrix)

  obj_points = [
    get3DCoordinates(img_points[0], inv_cam_matrix),
    get3DCoordinates(img_points[1], inv_cam_matrix)
  ]

  error = (
    obj_points[1][0] - obj_points[0][0],
    obj_points[1][1] - obj_points[0][1],
    obj_points[1][2] - obj_points[0][2]
  )

  print(
    """
    The back projected points have the following coordinates:

    (%.2f, %2.f) -> (%.2f, %.2f, %.2f)
    (%.2f, %2.f) -> (%.2f, %.2f, %.2f)

    The measured errors are:

    X axis: %.2f
    Y axis: %.2f
    Z axis: %.2f

    Euclidean norm of the error: %.2f""" % (
      img_points[0][0], img_points[0][1],
      obj_points[0][0], obj_points[0][1], obj_points[0][2],
      img_points[1][0], img_points[1][1],
      obj_points[1][0], obj_points[1][1], obj_points[1][2],
      math.fabs(obj_points[1][0] - obj_points[0][0]),
      math.fabs(obj_points[1][1] - obj_points[0][1]),
      math.fabs(obj_points[1][2] - obj_points[0][2]),
      math.sqrt(error[0]**2 + error[1]**2 + error[2]**2)
    )
 )


if __name__ == "__main__":
  parseArguments()
  runEstimation()
