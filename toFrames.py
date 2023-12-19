import cv2 as cv
import imutils

video = cv.VideoCapture('./Training_Data/jumps.MOV')
count = 1
while video.isOpened():
    ret, frame = video.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imwrite('./Training_Data/Frame_' + str(count) + '.jpg', frame)
    count += 1

video.release()
cv.destroyAllWindows()
