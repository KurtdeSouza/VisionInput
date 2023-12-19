import cv2
face_cascade = cv2.CascadeClassifier('./data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./data\haarcascades\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./data\haarcascades\haarcascade_smile.xml')


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        red=  (0, 0, 255)
        for (sx, sy, sw, sh) in smiles:
            im = cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), red, 2)
            cv2.putText(im, 'me when I see my gf', (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,red, 1)
    return frame
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('smile_detection', detect(gray,frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.imwrite('test_photo.jpg', frame)

cap.release()
cv2.destroyAllWindows()
