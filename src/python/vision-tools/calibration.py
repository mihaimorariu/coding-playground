import numpy as np
import cv2
import sys, getopt
import yaml

device = 0
rows = 6
cols = 8
size = 1.0
out_file = "./calibration.yaml"

def printUsage():
  print(
    """calibration.py [options]
    where [options] can be:

    -d - Device number or input stream (default: 0)
    -r - Number of calibration pattern rows (default: 6)
    -c - Number of calibration pattern cols (default: 8)
    -s - Square size (default: 1.0)
    -o - Output calibration file"""
  )

def parseArguments():
  global device, cols, rows, size, out_file

  try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:o:r:s:c:")
  except getopt.GetoptError:
    printUsage()
    sys.exit(2)

  for opt, arg in opts:
    if opt == "-h":
      printUsage()
    elif opt == "-d":
      try:
        device = int(arg)
      except ValueError:
        device = arg
    elif opt == "-o":
      out_file = arg
    elif opt == "-c":
      cols = int(arg)
    elif opt == "-r":
      rows = int(arg)
    elif opt == "-s":
      size = float(arg)
    else:
      printUsage()

def runCalibration():
  global device, rows, cols, size, out_file

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  pattern = np.zeros((rows * cols, 3), np.float32)
  pattern[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * size

  object_points = []
  image_points = []

  capture = cv2.VideoCapture(device)
  detection_on = False
  success = False

  cam_matrix = None
  dist_coeffs = None

  while True:
    _, image = capture.read()

    if detection_on:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      success, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

      if success:
        object_points.append(pattern)
        corners_refined = cv2.cornerSubPix(
          gray, corners, (7, 7), (-1, -1), criteria)
        image_points.append(corners_refined)

        detection_on = False
        cv2.drawChessboardCorners(image, (cols, rows), corners, success)

    if cam_matrix is not None:
      undistorted = cv2.undistort(image, cam_matrix, dist_coeffs)
      cv2.imshow("Camera [undistorted]", undistorted);

    cv2.putText(image, "d - Detect pattern", (30, 30),
      cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(image, "c - Calibrate camera", (30, 50),
      cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(image, "q - Quit", (30, 70),
      cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow("Camera [distorted]", image)

    if success:
      success = False
      cv2.waitKey(1000)
 
    key = cv2.waitKey(1) & 0xff

    if key == ord("q"):
      break
    elif key == ord("d"):
      detection_on = True
    elif key == ord("c"):
      if len(image_points) == 0:
        continue

      _, cam_matrix, dist_coeffs, rot_vecs, trans_vecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None)

      total_error = 0
      for i in range(len(object_points)):
        corners, _ = cv2.projectPoints(
          object_points[i], rot_vecs[i], trans_vecs[i], cam_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], corners, cv2.NORM_L2) / len(corners)
        total_error += error

      np.set_printoptions(precision = 3)
      np.set_printoptions(suppress = True)

      print("Camera matrix: \n", cam_matrix)
      print("Distortion coefficients: \n", dist_coeffs)
      print("Rotations: \n", rot_vecs)
      print("Translations: \n", trans_vecs)
      print("Reprojection error: ", total_error)

      with open(out_file, "w") as file:
        data = {
          "cam_matrix": cam_matrix.tolist(),
          "dist_coeffs": dist_coeffs.tolist(),
          "rot_vecs": rot_vecs,
          "trans_vecs": trans_vecs}
        file.write(yaml.dump(data, default_flow_style = True))

  cv2.destroyAllWindows()

if __name__ == "__main__":
  parseArguments()
  runCalibration()
