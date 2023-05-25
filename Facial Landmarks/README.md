# Facial Landmark Detection using dlib

  This code demonstrates how to use the dlib library to detect facial landmarks in real-time using a webcam. It uses the pre-trained shape predictor model from dlib to detect and locate 68 facial landmarks on detected faces.

## Prerequisites

  Before running the code, make sure you install the "requirements.txt" for necessary libraries.
  This code runs **Python 3.9.16**

  You also need to download the shape predictor model file "shape_predictor_68_face_landmarks.dat" from the dlib website. You can download it from the following link: 
[shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
 
  Extract the file and place it in the same directory as the code file.

## Usage
1. Clone or download the code file to your local machine.
2. Install the required dependencies using pip install -r requirements.txt 
3. Download the shape predictor model file from the link provided above and place it in the same directory as the code file.
4. Run the code using python <filename>.py.
5. A new window will open showing the webcam feed with facial landmarks detected in real-time.
6. Press the 'q' key to exit the program.

## Code Explanation
  The code begins by importing the necessary libraries: dlib, OpenCV, NumPy, and imutils.

  ```
  import dlib
  import cv2
  import numpy as np
  from imutils import face_utils  
  ```
  
  It then initializes the face detector and shape predictor using the pre-trained model.
  
  ```
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
  ```
  
  The code captures video frames from the webcam using OpenCV.
  ```
  cap = cv2.VideoCapture(0)
  ```
  
  Next, it retrieves the first frame and determines the size ratio to resize the image.
  ```
  ret, frame = cap.read()
  face_shape = frame.shape
  size_ratio = 10
  frame = cv2.resize(frame, (face_shape[1]//size_ratio, face_shape[0]//size_ratio))
  ```
  
  Inside the main loop, it continuously reads frames from the webcam.
  ```
  while True:
    ret, frame = cap.read()
  ```
  
  For each frame, it detects faces using the face detector.
  ```
  rects = detector(frame, 1)
  ```
  
  The code then loops over each detected face (in case of multiple faces in frame) and uses the shape predictor to estimate the facial landmarks.
  ```
  for (i, rect) in enumerate(rects):
    reference_points = predictor(frame, rect)
    reference_points = face_utils.shape_to_np(reference_points)
  ```
  
  It draws rectangles around the detected faces and overlays the facial landmarks on the frame.
  ```
  (x,y,w,h) = face_utils.rect_to_bb(rect)
  cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

  for (x,y) in reference_points:
      cv2.circle(frame, (x,y), 2, (0,255,0), -1)
  ```
  
  The modified frame with overlays is displayed in a separate window.
  ```
  cv2.imshow('face', frame)
  ```
  
  The program continues to run until the 'q' key is pressed, at which point it releases the camera and closes all windows.
  ```
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()
  ```



  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
