# Vehicle Tracking with YOLOV8

<img width="640" alt="output" src="https://github.com/thiagosmpa/Computer-Vision/assets/33949962/dc2594dd-93f9-4430-93f5-111b4066bb71">

This code used `YOLOV8` with native Tracking support to **Detect** vehicles, **Count** them and **Track** them on the roadway. Tracking tools can be used to give each object an unique ID, which will be used to follow this object through the next frames of the video. 

In addition, this implementation integrates `Supervision` tools, from Roboflow to provide robust support for annotation tasks within the images, thereby enhancing the accuracy and comprehensiveness of object detection and understanding.


## Prerequisites

Before running the script, make sure you have the following:

- Python 3.9.16
- Required packages: `ultralytics`, `opencv-python`, `supervision`, found in `requirements.txt` file.
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
