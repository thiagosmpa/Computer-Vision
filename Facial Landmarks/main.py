"""
    Using dlib to detect facial landmarks via webcam image
""" 

import dlib
import cv2
import numpy as np
from imutils import face_utils

# start detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# download predictor from link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# start camera
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    size = frame.shape
    aspect_ratio = 1

    # resize frame with aspect ratio
    frame = cv2.resize(frame, (size[1]//aspect_ratio, size[0]//aspect_ratio))
    
    # detect faces
    rects = detector(frame, 0)

    # for loop is used to detect multiple faces in the frame
    for (i, rect) in enumerate(rects):
        reference_points = predictor(frame, rect)
        reference_points = face_utils.shape_to_np(reference_points)
        
        (x,y,w,h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        for (x,y) in reference_points:
            cv2.circle(frame, (x,y), 2, (0,255,0), -1)
        
    cv2.imshow('face', frame)

    # if q is pressed, break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# release camera and close all windows
cap.release()
cv2.destroyAllWindows()







