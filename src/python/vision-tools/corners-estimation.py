import cv2
import yaml
import numpy as np
import math, sys

frame = None
translation_wc = None
rotation_wc = None
rotation_cw = None

cam_matrix = None
inv_cam_matrix = None
dist_coeffs = None

img_points = []
obj_size = None

def printUsage():
  print(
    "corners-estimation.py [calib_file] [dev_number] [width] [height] [depth]"
  )

def mouseCallback(event, x, y, flags, param):
  global frame, rotation_cw, inv_cam_matrix
  global translation_wc, img_points, obj_size
  global rotation_wc, dist_coeffs

  if event == cv2.EVENT_LBUTTONDOWN:
    img_points.append((x, y))

    if len(img_points) == 4:
      center = rotation_cw * (-translation_wc)
      reprojections = np.zeros((4, 3), np.float32)

      directions = []
      for i in range(len(img_points)):
        point = np.matrix([img_points[i][0], img_points[i][1], 1])
        reproj = rotation_cw * (inv_cam_matrix * point.T - translation_wc)

        direction = reproj - center
        directions.append(direction)

      coeffs = np.matrix([
        [-directions[0][0, 0], directions[1][0, 0], 0, 0],
        [0, 0, directions[2][0, 0], -directions[3][0, 0]],
        [-directions[0][1, 0], 0, 0, directions[3][1, 0]],
        [0, -directions[1][1, 0], directions[2][1, 0], 0],
        [directions[0][2, 0], -directions[1][2, 0], 0, 0],
        [0, directions[1][2, 0], -directions[2][2, 0], 0],
        [0, 0, directions[2][2, 0], -directions[3][2, 0]],
      ])

      width, height, depth = obj_size
      sizes = np.matrix([width, width, height, height, 0, 0, 0]).T
      lambdas, _, _, _ = np.linalg.lstsq(coeffs, sizes)

      obj_points = []
      for i in range(len(lambdas)):
        obj_points.append(center + lambdas[i, 0] * directions[i])

      print("\nBack projections: ")
      for point in obj_points:
        print("(%.2f, %.2f, %.2f)" % (point[0, 0], point[1, 0], point[2, 0]))

      mean_points = np.mean(obj_points, axis = 0)
      box_center = np.array([
        [mean_points[0, 0], mean_points[1, 0], mean_points[2, 0]]
      ])
      proj_center, _ = cv2.projectPoints(
        box_center, rotation_wc, translation_wc, cam_matrix, dist_coeffs
      )
      x, y = proj_center[0][0]
      cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 200), -1)

      corners_projections = []
      for point in obj_points:
        obj_corners = np.array([
          [point[0, 0], point[1, 0], point[2, 0]],
          [point[0, 0], point[1, 0], point[2, 0] + depth]
        ])

        corners, _ = cv2.projectPoints(
          obj_corners, rotation_wc, translation_wc, cam_matrix, dist_coeffs
        )

        corners_projections.append(corners[0])
        corners_projections.append(corners[1])

      for corner in corners_projections:
        x, y = corner[0]
        cv2.circle(frame, (int(x), int(y)), 2, (0, 200, 0), -1)

      cv2.imshow("Camera", frame)

def runEstimation():
  global frame, rotation_cw
  global cam_matrix, inv_cam_matrix, dist_coeffs
  global rotation_wc, translation_wc
  global img_points, obj_size

  if len(sys.argv) != 6:
    printUsage()
    exit(0)

  with open(sys.argv[1], "r") as f:
    data = yaml.load(f)

  try:
    device = int(sys.argv[2])
  except ValueError:
    device = sys.argv[2]

  try:
    width = float(sys.argv[3])
    height = float(sys.argv[4])
    depth = float(sys.argv[5])
  except ValueError:
    printUsage()
    exit(0)

  np.set_printoptions(precision = 3)
  np.set_printoptions(suppress = True)

  obj_size = (width, height, depth)
  cam_matrix = np.matrix(data["cam_matrix"])
  dist_coeffs = np.matrix(data["dist_coeffs"])
  rotation_wc = np.matrix(data["rot_vecs"][-1])
  translation_wc = np.matrix(data["trans_vecs"][-1])
  rotation_wc, _ = cv2.Rodrigues(rotation_wc)

  print("Camera matrix: \n", cam_matrix)
  print("Distortion coefficients: \n", dist_coeffs)
  print("Translation (world to camera): \n", translation_wc)
  print("Rotation (world to camera): \n", rotation_wc)

  inv_cam_matrix = np.linalg.inv(cam_matrix)
  rotation_cw = np.linalg.inv(rotation_wc)

  capture = cv2.VideoCapture(device)
  _, frame = capture.read()
  frame = cv2.undistort(frame, cam_matrix, dist_coeffs)

  cv2.namedWindow("Camera")
  cv2.setMouseCallback("Camera", mouseCallback)

  cv2.imshow("Camera", frame)
  cv2.waitKey(0)

if __name__ == "__main__":
  runEstimation()

