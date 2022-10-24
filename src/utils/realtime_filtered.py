from wsgiref.util import setup_testing_defaults
from cv2 import normalize
import numpy as np 
import cv2
 
# Check for left and right camera IDs
# These values can change depending on the system
CamL_id = 2 # Camera ID for left camera
CamR_id = 0 # Camera ID for right camera
 
CamL= cv2.VideoCapture(CamL_id)
CamR= cv2.VideoCapture(CamR_id)

CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
CamL.set(cv2.CAP_PROP_AUTOFOCUS, 0)

CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
CamR.set(cv2.CAP_PROP_AUTOFOCUS, 0)
 
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("resources/stereoMap.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode('stereoMapL_x').mat()
Left_Stereo_Map_y = cv_file.getNode('stereoMapL_y').mat()
Right_Stereo_Map_x = cv_file.getNode('stereoMapR_x').mat()
Right_Stereo_Map_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()
 
def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)

# create trackbars
cv2.createTrackbar('minDisparity','disp', 32, 50, nothing)
cv2.createTrackbar('numDisparities','disp', 5, 32, nothing)
cv2.createTrackbar('blockSize','disp', 5, 50, nothing)

cv2.createTrackbar('disp12MaxDiff','disp', 5, 25, nothing)
cv2.createTrackbar('preFilterCap','disp', 5, 62, nothing)
cv2.createTrackbar('uniquenessRatio','disp', 10, 100, nothing)
cv2.createTrackbar('speckleWindowSize','disp', 100, 200, nothing)
cv2.createTrackbar('speckleRange','disp', 32, 100, nothing)
cv2.createTrackbar('mode','disp', 0, 3, nothing)
 
sigma = 1.5 # Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. 
lmbda = 8000.0 # Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000
left_matcher = cv2.StereoSGBM_create()
 
while True:
 
  # Capturing and storing left and right camera images
  retL, imgL= CamL.read()
  retR, imgR= CamR.read()
  
   
  # Proceed only if the frames have been captured
  if retL and retR:
    # imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    # imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    imgR_gray = imgR
    imgL_gray = imgL
 
    # Applying stereo image rectification on the left image
    Left_nice= cv2.remap(imgL_gray,
              Left_Stereo_Map_x,
              Left_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
     
    # Applying stereo image rectification on the right image
    Right_nice= cv2.remap(imgR_gray,
              Right_Stereo_Map_x,
              Right_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)


    # Updating the parameters based on the trackbar positions
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    mode = cv2.getTrackbarPos('mode', 'disp')

    # Setting the updated parameters before computing disparity map
    left_matcher.setMinDisparity(minDisparity)
    left_matcher.setNumDisparities(numDisparities)
    left_matcher.setBlockSize(blockSize)
    left_matcher.setP1(8*1*blockSize*blockSize) 
    left_matcher.setP2(32*1*blockSize*blockSize)
    left_matcher.setDisp12MaxDiff(disp12MaxDiff)
    left_matcher.setPreFilterCap(preFilterCap)
    left_matcher.setUniquenessRatio(uniquenessRatio)
    left_matcher.setSpeckleWindowSize(speckleWindowSize)
    left_matcher.setSpeckleRange(speckleRange)
    left_matcher.setMode(mode)
 
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    left_disp = left_matcher.compute(Left_nice, Right_nice)
    right_disp = right_matcher.compute(Right_nice, Left_nice)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(left_disp, Left_nice, disparity_map_right=right_disp)

    # Converting to float32 
    filtered_disp = filtered_disp.astype(np.float32)
    # Scaling down the disparity values and normalizing them 
    filtered_disp = (filtered_disp/16.0 - left_matcher.getMinDisparity())/left_matcher.getNumDisparities()
 
    normalized_disp = cv2.normalize(filtered_disp, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    print(normalized_disp.max())
    print(normalized_disp.min())

    # focalLength = 627.93
    # baseLine = 200     
    # depth = baseLine * focalLength / filtered_disp

    coloredDepth = cv2.applyColorMap(normalized_disp, cv2.COLORMAP_JET)

    # Displaying the disparity map
    cv2.imshow("disp", filtered_disp)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break
   
  else:
    CamL= cv2.VideoCapture(CamL_id)
    CamR= cv2.VideoCapture(CamR_id)