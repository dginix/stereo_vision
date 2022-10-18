import numpy as np
import cv2
import glob
import argparse
import sys


def showImgMonoCalibrate(origImg, C, D, imgShape, name):
    newcameramtx, testRoi = cv2.getOptimalNewCameraMatrix(C, D, imgShape, 1, imgShape)

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(C, D, None, newcameramtx, imgShape, 5)
    result = cv2.remap(origImg, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    # x, y, w, h = testRoi
    # result = result[y:y+h, x:x+w]
    cv2.imshow('original {name}', origImg)
    cv2.imshow('remap {name}', result)



def stereo_calibrate(right_dir, left_dir, image_format = "jpg", square_size=0.03, width=7, height=5):

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

    imgShape = cv2.imread(left_images[0]).shape[1::-1]

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
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        left = cv2.imread(left_img)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret_left and ret_right:  # If both image is okay. Otherwise we explain which pair has a problem and continue
            # Object points
            objpoints.append(objp)

            # Right points
            corners_right2 = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            imgpoints_right.append(corners_right2)

            # Left points
            corners_left2 = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            imgpoints_left.append(corners_left2)

            good_count += 1
        else:
            print("Chessboard couldn't detected. Image pair: ", left_img, " and ", right_img)
            err_count += 1
            continue

    print(good_count, err_count)

    mse1, C1, D1, R1, T1 = cv2.calibrateCamera(objpoints, imgpoints_left, imgShape, 
                        None, None, flags=0)
    mse2, C2, D2, R2, T2 = cv2.calibrateCamera(objpoints, imgpoints_right, imgShape, 
                        None, None, flags=0)


    testImgLeft = cv2.imread('./resources/calibration_images/LEFT/l_1.jpg')
    testImgRight = cv2.imread('./resources/calibration_images/RIGHT/r_1.jpg')

    #show new images
    showImgMonoCalibrate(testImgLeft, C1, D1, imgShape, "left")

    stereoFlags = 0
    #stereoFlags |= cv2.CALIB_FIX_INTRINSIC
    #stereoFlags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    #stereoFlags |= cv2.CALIB_USE_INTRINSIC_GUESS
    #stereoFlags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    #stereoFlags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                            cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    mseTotal, CL, DL, CR, DR, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, C1, D1, C2, D2, imgShape, 
        flags = stereoFlags, criteria=stereocalib_criteria)

    RL, RR, PL, PR, Q, validROIL, validROIR = cv2.stereoRectify(CL, DL, CR, DR, imgShape, R, T, alpha=0.8)

    undistL, rectifL = cv2.initUndistortRectifyMap(CL, DL, RL, PL, imgShape, cv2.CV_32FC1)
    undistR, rectifR = cv2.initUndistortRectifyMap(CR, DR, RR, PR, imgShape, cv2.CV_32FC1)

    trueImgLeft = cv2.remap(testImgLeft, undistL, rectifL, cv2.INTER_LINEAR)
    trueImgRight = cv2.remap(testImgRight, undistR, rectifR, cv2.INTER_LINEAR)

    conc_frame = np.concatenate((trueImgLeft, trueImgRight), axis=1)
    cv2.imshow('Camera calibration', conc_frame)
    cv2.waitKey(0)


right_coef_path = "./calibration_data/right_coefs/right.yml"
left_coef_path = "./calibration_data/left_coefs/left.yml"
stereo_coef_path = "./calibration_data/stereo_coefs/stereo.yml"

right_imgs_path = "./resources/calibration_images/LEFT"
left_imgs_path = "./resources/calibration_images/LEFT"

square_size = 0.03
chess_width = 7
chess_height = 5

stereo_calibrate(right_dir=right_imgs_path, left_dir=left_imgs_path)


