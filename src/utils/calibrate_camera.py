import numpy as np
import cv2
import glob
import argparse
import sys


def stereo_calibrate(right_dir, left_dir, image_format, square_size=0.03, width=7, height=5):
    # !!! Add coefficient saving
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pattern_size = (width, height)  # Chessboard size!

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords. Use your metric.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []  # 2d points in image plane.

    err_count = 0  # chessboard not founded
    good_count = 0  # chessboard founded

    if left_dir[-1:] == '/':
        left_dir = left_dir[:-1]

    if right_dir[-1:] == '/':
        right_dir = right_dir[:-1]

    left_images = glob.glob(left_dir + '/' + '*.' + image_format)
    right_images = glob.glob(right_dir + '/' + '*.' + image_format)

    left_images.sort()
    right_images.sort()

    imgShape = cv2.imread(left_images[0]).shape[:2:]

    print(imgShape)

    if len(left_images) != len(right_images):
        print("Numbers of left and right images are not equal. They should be pairs.")
        print("Left images count: ", len(left_images))
        print("Right images count: ", len(right_images))
        sys.exit(-1)

    pair_images = zip(left_images, right_images)

    for left_img, right_img in pair_images:
        right = cv2.imread(right_img)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        left = cv2.imread(left_img)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:  # If both image is okay. Otherwise we explain which pair has a problem and continue
            # Object points
            objpoints.append(objp)
            # Right points
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            imgpoints_right.append(corners_right)
            # Left points
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            imgpoints_left.append(corners_left)
            good_count += 1
        else:
            print("Chessboard couldn't detected. Image pair: ", left_img, " and ", right_img)
            err_count += 1
            continue

    _, mtxLeft, distLeft, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, imgShape, None, None)
    _, mtxRight, distRight, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, imgShape, None, None)

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, imgpoints_left, imgpoints_right, mtxLeft, distLeft,
                                                          mtxRight, distRight, imgShape)

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, imgShape, R, T,
                                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)


    # mtx = K
    # dist = D

    print(good_count, err_count)



right_coef_path = "./calibration_data/right_coefs/right.yml"
left_coef_path = "./calibration_data/left_coefs/left.yml"
stereo_coef_path = "./calibration_data/stereo_coefs/stereo.yml"

right_imgs_path = "./calibration_data/calibration_images/RIGHT"
left_imgs_path = "./calibration_data/calibration_images/LEFT"

image_format = "jpg"
square_size = 0.03

chess_width = 7
chess_height = 5

stereo_calibrate(right_dir=right_imgs_path, left_dir=left_imgs_path, image_format="jpg")


