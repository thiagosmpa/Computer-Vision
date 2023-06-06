#!/usr/bin/env python3
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from drowsiness_detect import eyes_characteristics, mouth_characteristics

# start detector and predictor
# download predictor from link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_location = '../shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_location)

eyeChar = eyes_characteristics()
mouthChar = mouth_characteristics()

ear_values = []
mar_values = []


frame_counter_eyes = 0
ear_threshold = 0.22
drowsiness_frames_threshold = 2

frame_counter_mouth = 0
mar_threshold = 0.85
yawn_frames_threshold = 1

fig, axs = plt.subplots(2, 1)


# Start video stream (live videostream in blank for a webcam recording)
cap = VideoStream('src/video.MOV').start()

while True:
    frame = cap.read()
    if frame is None:
        break
    
    frame = imutils.resize(frame, width=1024)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    for (i, rect) in enumerate(rects):
        
        # get face landmarks
        reference_points = predictor(frame, rect)
        reference_points = face_utils.shape_to_np(reference_points)
        
        # draw bounding box on face
        # cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
        
        # get eyes and mouth
        left_eye, right_eye = eyeChar.get_eyes(reference_points)
        mouth = mouthChar.get_mouth(reference_points)
        
        left_eyeHull = cv2.convexHull(left_eye)
        right_eyeHull = cv2.convexHull(right_eye)
        mouthHull = cv2.convexHull(mouth)
        # Hull is used to draw the contours of the eyes and mouth
        
        # Draw face, eyes and mouth
        for (x, y) in reference_points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        cv2.drawContours(frame, [left_eyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eyeHull], -1, (0, 255, 0), 1)
        
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        left_eye_ear = eyeChar.get_ear(left_eye)
        right_eye_ear = eyeChar.get_ear(right_eye)
        eye_ear = (left_eye_ear + right_eye_ear) / 2
        
        mouth_mar = mouthChar.get_mar(mouth)
        
        ear_values.append(eye_ear)
        mar_values.append(mouth_mar)
        
        # Set Drowsiness and Yawn Alert
        if eye_ear < ear_threshold:
            frame_counter_eyes = frame_counter_eyes + 1
            if frame_counter_eyes > drowsiness_frames_threshold:
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            frame_counter_eyes = 0
        
        # set counter for how many frames mouth is open
        if mouth_mar > mar_threshold:
            frame_counter_mouth = frame_counter_mouth + 1
            if frame_counter_mouth > yawn_frames_threshold:
                cv2.putText(frame, "YAWN ALERT!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            frame_counter_mouth = 0
        
        
        # Prepare data for plotting real time
        ear_data = ear_values[-30:]
        mar_data = mar_values[-30:]
        
        # EAR plot
        axs[0].clear()
        axs[0].set_title('Eyes Aspect Ratio (EAR) over the last 30 frames')
        axs[0].set_ylabel('EAR')
        axs[0].set_ylim([0, 0.5])
        axs[0].get_xaxis().set_visible(False)
        
        ear_line, = axs[0].plot(range(len(ear_data)), ear_data, 'r')
        axs[0].plot([0, 30], [ear_threshold, ear_threshold], 'b--')
        axs[0].text(0, ear_threshold + 0.02, 'Threshold', fontsize=10)

        # MAR plot
        axs[1].clear()
        axs[1].set_title('Mouth Aspect Ratio (MAR) over the last 30 frames')
        axs[1].set_xlabel('Time (frames)')
        axs[1].set_ylabel('MAR')
        axs[1].set_ylim([0.5, 1])
        
        mar_line, = axs[1].plot(range(len(mar_data)), mar_data, 'r')
        axs[1].plot([0, 30], [mar_threshold, mar_threshold], 'b--')
        axs[1].text(0, mar_threshold + 0.02, 'Threshold', fontsize=10)

        fig.canvas.draw()
        plt.show(block=False)

        ear_line.set_data(range(len(ear_data)), ear_values[-30:])
        mar_line.set_data(range(len(mar_data)), mar_values[-30:])
        
        plt.pause(0.1)
        fig.canvas.draw()
        
        
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.stop()

# save plot to file
fig.savefig('plot.png')
# save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1024, 576))

for i in range(len(ear_values)):
    frame = cv2.imread('plot.png')
    frame = cv2.resize(frame, (1024, 576))
    out.write(frame)
out.release()
