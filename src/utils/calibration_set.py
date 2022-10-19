import cv2
import numpy as np
import os
import glob


def init_text_field(frame):
    height, width = frame.shape[:2]
    blank_image = np.zeros((height, 300, 3), np.uint8)
    blank_image[:] = (255, 255, 255)

    text = "[SPACE] - take photo"
    cv2.putText(blank_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1, cv2.LINE_AA)

    text = "[c] - clear set"
    cv2.putText(blank_image, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1, cv2.LINE_AA)

    text = "[l] - delete last"
    cv2.putText(blank_image, text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1, cv2.LINE_AA)

    return blank_image


def change_text_field(frame):
    text = "Photo count: " + str(photo_count)
    cv2.putText(frame, text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1, cv2.LINE_AA)
    return frame


photo_count = 0
left_img_filepath = 'resources/calibration_images/LEFT/' 
right_img_filepath = 'resources/calibration_images/RIGHT/'

if not os.path.exists(left_img_filepath):
    os.mkdir(left_img_filepath)

if not os.path.exists(right_img_filepath):
    os.mkdir(right_img_filepath)

cv2.namedWindow('Camera calibration')

cam_right = cv2.VideoCapture(0)
cam_left = cv2.VideoCapture(2)

cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

check, frame_right = cam_right.read()
text_filed = init_text_field(frame_right)

while True:
    check, frame_left = cam_left.read()
    check, frame_right = cam_right.read()

    cv2.putText(frame_left, 'L', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame_right, 'R', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1, cv2.LINE_AA)

    conc_frame = np.concatenate((frame_left, frame_right, text_filed), axis=1)

    cv2.imshow('Camera calibration', conc_frame)
    key = cv2.waitKey(1)

    if key == ord(' '):
        # take photo
        photo_count += 1
        text_filed = init_text_field(text_filed)
        change_text_field(text_filed)

        filename = left_img_filepath + 'l_' + str(photo_count) + '.jpg'
        cv2.imwrite(filename, frame_left)

        filename = right_img_filepath + 'r_' + str(photo_count) + '.jpg'
        cv2.imwrite(filename, frame_right)

    elif key == ord('c'):
        # clear photo set
        files_l = glob.glob(left_img_filepath + '*')
        files_r = glob.glob(right_img_filepath + '*')

        for file in files_l:
            os.remove(file)

        for file in files_r:
            os.remove(file)

        photo_count = 0
        text_filed = init_text_field(text_filed)
        change_text_field(text_filed)

    elif key == ord('l'):
        # delete last
        list_of_files_l = glob.glob(left_img_filepath + '*') # * means all if need specific format then *.csv
        latest_file_l = max(list_of_files_l, key=os.path.getctime)
        os.remove(latest_file_l)

        list_of_files_r = glob.glob(right_img_filepath + '*') # * means all if need specific format then *.csv
        latest_file_r = max(list_of_files_r, key=os.path.getctime)
        os.remove(latest_file_r)

        photo_count -= 1
        text_filed = init_text_field(text_filed)
        change_text_field(text_filed)


    elif key == 27:
        break

cam_left.release()
cam_right.release()
cv2.destroyAllWindows()