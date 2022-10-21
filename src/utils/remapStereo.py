import sys
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('resources/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

frameL = cv2.imread('resources/testL.png')
frameR = cv2.imread('resources/testR.png')

# undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
# undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# cv2.imshow('undistortedR', undistortedL)

# conc_frame = np.concatenate((undistortedL, undistortedR), axis=1)
# for i in range(0, conc_frame.shape[0], int(conc_frame.shape[0]/15)):
#     cv2.line(conc_frame, (0,i), (conc_frame.shape[1],i), (255,0,0), 1)
# cv2.imshow('Camera calibration', conc_frame)
# cv2.waitKey(0)

# stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=25)

# grayL = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
# grayR = cv2.cvtColor(undistortedR, cv2.COLOR_BGR2GRAY)
# disparity = stereo.compute(grayL, grayR)

# plt.imshow(disparity)
# plt.show()

#========================

cam_right = cv2.VideoCapture(0)
cam_left = cv2.VideoCapture(2)

stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=25)

while True:

    check, frame_left = cam_left.read()
    check, frame_right = cam_right.read()

    undistortedL= cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    grayL = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(undistortedR, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR)

    plt.imshow(disparity)
    plt.show()

    key = cv2.waitKey(1)
    if key == 27:
        break

cam_left.release()
cam_right.release()
cv2.destroyAllWindows()