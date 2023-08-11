#!/usr/bin/env python

from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from trackingtools import track_vehicles
import os

source = './src/video.mp4'

LINE_START = sv.Point(0, 450)
LINE_END = sv.Point(1280, 450)

def main():
    print('Starting main...')
    
    if not os.path.exists('./output'):
        os.mkdir('./output')
    output_filename = './output/output_video.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec ('mp4v' for MP4)
    out = cv2.VideoWriter(output_filename, fourcc, 25.0, (1280, 720))  # Video is written in 25 FPS, 1280x720
    
    model = YOLO('yolov8s.pt')
    line_zone = sv.LineZone(LINE_START, LINE_END)
    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    
    for result in model.track(source=source, stream=True):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        
        # Filter detections using classes: 2: car, 5: bus, 7: truck
        filter_condition = np.isin(detections.class_id, [2, 5, 7])
        detections = detections[filter_condition]
        
        # This is because the program crashes when theres no tracker ids in the screen
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:.2f}" for tracker_id, class_id, confidence 
                in zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]
        else:
            labels = [
                f"{model.model.names[class_id]} {confidence:.2f}" for class_id, confidence 
                in zip(detections.class_id, detections.confidence)
            ]
        
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
        
        line_zone.trigger(detections=detections)
        # in_count = line_zone.in_count
        out_count = line_zone.out_count
        
        cv2.line(frame, (0,450), (1280, 450), (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles counting: {out_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # load track_vehicles from trackingtools.py
        track_vehicles(detections, frame)
        
        out.write(frame)
        
        cv2.imshow("yolov8", frame)
        if (cv2.waitKey(30) == 27):
            break
        
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
