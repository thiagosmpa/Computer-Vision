# Drowsiness Detection using DLib

<img width="640" alt="Captura de Tela 2023-08-11 às 10 35 48" src="https://github.com/thiagosmpa/Computer-Vision/assets/33949962/2111806d-00d1-4d97-921e-d532788bd6cf">


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

