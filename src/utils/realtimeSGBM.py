from wsgiref.util import setup_testing_defaults
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

CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
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
#cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('minDisparity','disp', 32, 50, nothing)
cv2.createTrackbar('numDisparities','disp', 5, 32, nothing)
cv2.createTrackbar('blockSize','disp', 5, 50, nothing)

cv2.createTrackbar('disp12MaxDiff','disp', 5, 25, nothing)
cv2.createTrackbar('preFilterCap','disp', 5, 62, nothing)
cv2.createTrackbar('uniquenessRatio','disp', 10, 100, nothing)
cv2.createTrackbar('speckleWindowSize','disp', 100, 200, nothing)
cv2.createTrackbar('speckleRange','disp', 32, 100, nothing)
cv2.createTrackbar('mode','disp', 0, 3, nothing)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM_create()
 
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
    stereo.setMinDisparity(minDisparity)
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setP1(8*3*blockSize*blockSize) 
    stereo.setP2(32*3*blockSize*blockSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setSpeckleRange(speckleRange)
    stereo.setMode(mode)

    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    cv2.imshow("disp",disparity)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break
   
  else:
    CamL= cv2.VideoCapture(CamL_id)
    CamR= cv2.VideoCapture(CamR_id)