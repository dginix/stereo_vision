import cv2


capL = cv2.VideoCapture(0) # L
capR = cv2.VideoCapture(2) # R


capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

num = 0

while capL.isOpened():

    _, imgL = capL.read()
    _, imgR = capR.read()


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('resources/calibration_images/LEFT/imageL' + str(num) + '.png', imgL)
        cv2.imwrite('resources/calibration_images/RIGHT/imageR' + str(num) + '.png', imgR)
        print("images saved!")
        num += 1

    cv2.imshow('Img Left',imgL)
    cv2.imshow('Img Right',imgR)
