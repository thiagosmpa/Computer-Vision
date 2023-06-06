from imutils import face_utils
from scipy.spatial import distance as dist

class eyes_characteristics ():
    
    def get_eyes(self, reference_points):
        # this could be achieved by using the predictor points, but this is easier
        (begin_left, end_left) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (begin_right, end_right) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        left_eye = reference_points[begin_left:end_left]
        right_eye = reference_points[begin_right:end_right]
        
        return left_eye, right_eye
    
    def get_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        
        ear = (A+B)/(2.0*C)
        return ear
    
    
class mouth_characteristics():
    
    def get_mouth(self, reference_points):
        (begin, end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        mouth = reference_points[begin:end]
        return mouth
    
    def get_mar(self, mouth):
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
        D = dist.euclidean(mouth[0], mouth[4])
        
        mar = (A+B+C)/(3.0*D)
        return mar
