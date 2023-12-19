import cv2
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('user_cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.imwrite('test_photo.jpg', frame)

cap.release()
cv2.destroyAllWindows()