import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_frame(frame, gray):
    faces = face_classifier.detectMultiScale(gray, 1.3, 2)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_frame, (ex,ey), (ex+ew, ey+eh), (255,0,255), 2)
    return frame


video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect_frame(frame, gray)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
