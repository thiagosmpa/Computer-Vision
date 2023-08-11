"""
track_vehicles is used to draw tracking points for vehicles. In sum, this will be used to estimate the speed of vehicles
"""

import numpy as np
import cv2

TRACKING_LENGTH = 15
tracking = {}

def center(rectangle):
    xmin = rectangle[0]
    xmax = rectangle[2]
    ymax = rectangle[3]
    
    center_x = (int((xmin + xmax) / 2))
    center_y = (int(ymax))
    
    return (center_x, center_y)

def calculate_distance():
    return

def track_vehicles(detections, source):
    #check if detections.tracker_id is in tracking dictionary
    for tracker in detections.tracker_id:
        
        # The try / catch will avoid possible index errors
        try:
            indices = np.where(detections.tracker_id == tracker)[0]
            if len(indices) > 0:
                box = detections.xyxy[indices[0]]
            
            if tracker not in tracking:
                tracking[tracker] = [center(box)]
            else:
                if len(tracking[tracker]) < TRACKING_LENGTH:
                    tracking[tracker].append(center(box))
                else:
                    tracking[tracker].append(center(box))
                    tracking[tracker].pop(0)
                    
            # draw tracking points
            for point in tracking[tracker]:
                cv2.circle(source, point, 3, (0, 0, 255), -1)
        
        except IndexError:
            pass