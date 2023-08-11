# Vehicle Tracking with YOLO and Line Detection

This repository contains a Python script that utilizes the YOLO object detection model for vehicle tracking. It employs the Ultralytics YOLO library, OpenCV, and custom utilities for tracking and supervision.

## Prerequisites

Before running the script, make sure you have the following:

- Python 3.6 or later
- Required packages: `ultralytics`, `opencv-python`, `supervision` from Roboflow and custom packages (`trackingtools`) which you can include by installing them or placing them in the same directory as the script.
- Pre-trained YOLO model weights (`yolov8s.pt`), which will be downloaded automatically when loaded. 

## Usage

The script performs the following steps:

1. Initializes the YOLO model with the provided pre-trained weights.
2. Sets up a line zone for vehicle counting.
3. Iterates over each frame in the video, performing object detection.
4. Filters the detected objects to include only vehicles (cars, buses, and trucks).
5. Annotates the frame with detection information and vehicle counting.
6. Displays the annotated frame in real-time.
7. Tracks the detected vehicles using custom tracking tools.
8. Outputs a new video file with annotated frames.

The `output` directory will contain the resulting annotated video.

## Customization

You can modify the following parameters in the script:

- `LINE_START` and `LINE_END` for adjusting the counting line position.
- Annotator settings like thickness, text scale, and text thickness.
- Object classes to consider for tracking (e.g., cars, buses, trucks).

Feel free to customize the script to suit your specific use case.

## Acknowledgments

This script was developed using the Ultralytics YOLO library, OpenCV, and custom utilities for object tracking and supervision.
