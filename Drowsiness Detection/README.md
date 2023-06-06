# Drowsiness and Yawn Detection using Facial Landmarks

This project utilizes facial landmarks and computer vision techniques to detect drowsiness and yawns in real-time. It employs the dlib library for face detection and facial landmark localization. The program analyzes the eye aspect ratio (EAR) and mouth aspect ratio (MAR) to determine if the person is experiencing drowsiness or yawning.

## Requirements

- Python 3.x
- OpenCV
- dlib
- imutils
- numpy
- matplotlib

## Setup

1. Install the required dependencies mentioned above.
2. Download the `shape_predictor_68_face_landmarks.dat` file from the dlib website and place it in the project directory.
3. Adjust the `predictor_location` variable in the code to point to the correct location of the `shape_predictor_68_face_landmarks.dat` file.

## Usage

1. Run the script using `python your_script_name.py`.
2. The script will start the video stream (you can modify the source to use a different video file or a webcam).
3. The script will analyze each frame, detect faces, and calculate the eye and mouth characteristics.
4. If drowsiness is detected (eye aspect ratio below the threshold), a "DROWSINESS ALERT!" message will be displayed on the frame.
5. If a yawn is detected (mouth aspect ratio above the threshold), a "YAWN ALERT!" message will be displayed on the frame.
6. The script will also display real-time plots of the eye aspect ratio (EAR) and mouth aspect ratio (MAR) over the last 30 frames.
7. The plots will be saved as `plot.png`, and the processed video with alerts will be saved as `output.avi` in the project directory.

