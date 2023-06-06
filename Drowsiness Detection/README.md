# Facial Landmark Detection using dlib

This code demonstrates how to use the dlib library to detect facial landmarks in real-time using a webcam. It uses the pre-trained shape predictor model from dlib to detect and locate 68 facial landmarks on detected faces.

## Prerequisites

Before running the code, make sure you have the following prerequisites:

- Python 3.9.16 installed
- Install the required dependencies using `pip install -r requirements.txt`
- Download the shape predictor model file "shape_predictor_68_face_landmarks.dat" from the dlib website: [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract the file and place it in the same directory as the code file.

## Usage

1. Clone or download the code file to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the shape predictor model file from the link provided above and place it in the same directory as the code file.
4. Run the code using `python <filename>.py`.
5. A new window will open showing the webcam feed with facial landmarks detected in real-time.
6. Press the 'q' key to exit the program.

## Code Explanation

The code begins by importing the necessary libraries: dlib, OpenCV, NumPy, matplotlib, and imutils.

```
python
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from drowsiness_detect import eyes_characteristics, mouth_characteristics
```

Next, the code initializes the face detector and shape predictor using the pre-trained model.

```
predictor_location = '../shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_location)
```

The code also sets up the necessary variables and objects for drowsiness detection and mouth characteristics.

Inside the main loop, the code continuously reads frames from the webcam and performs facial landmark detection.

```
cap = VideoStream('src/video.MOV').start()

while True:
    frame = cap.read()
    if frame is None:
        break

    # Preprocess frame and convert to grayscale
    frame = imutils.resize(frame, width=1024)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the face detector
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        # Get face landmarks
        reference_points = predictor(frame, rect)
        reference_points = face_utils.shape_to_np(reference_points)

        # Draw bounding box on face
        # cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

        # Get eyes and mouth
        left_eye, right_eye = eyeChar.get_eyes(reference_points)
        mouth = mouthChar.get_mouth(reference_points)

        # Draw contours of eyes and mouth
        # ...

        # Calculate eye aspect ratio (EAR) and mouth aspect ratio (MAR)
        # ...

        # Set drowsiness and yawn alerts
        # ...

        # Prepare data for real-time plotting
        # ...

        # Display frame with overlays
        cv2.imshow('frame', frame)

        # Check for 'q' key press to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up and release resources
cv2.destroyAllWindows()
cap.stop()

# Save plot to file
fig.savefig('plot.png')

# Save video
# ...
```

You can customize the code further based on your specific requirements. Make sure to provide the correct video source and adjust any thresholds or parameters as needed.
