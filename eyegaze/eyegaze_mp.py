import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/ajeet/Downloads/shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_eye_region(landmarks, eye_points):
    left = landmarks.part(eye_points[0]).x
    right = landmarks.part(eye_points[3]).x
    top = min(landmarks.part(eye_points[1]).y, landmarks.part(eye_points[2]).y)
    bottom = max(landmarks.part(eye_points[5]).y, landmarks.part(eye_points[4]).y)
    return left, top, right, bottom

def get_gaze_ratio(eye_points, facial_landmarks, frame_gray):
    left, top, right, bottom = get_eye_region(facial_landmarks, eye_points)
    eye_roi = frame_gray[top:bottom, left:right]
    _, threshold_eye = cv2.threshold(eye_roi, 55, 255, cv2.THRESH_BINARY_INV)

    moments = cv2.moments(threshold_eye)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
    else:
        cx = (right - left) // 2 

    gaze_ratio = cx / (right - left) 
    return gaze_ratio

def get_gaze_direction(left_ratio, right_ratio):
    """Determine gaze direction based on gaze ratios"""
    avg_ratio = (left_ratio + right_ratio) / 2
    if avg_ratio < 0.4:
        return "Looking Right"
    elif avg_ratio > 0.6:
        return "Looking Left"
    else:
        return "Looking Center"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray)
        right_eye_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)

        gaze_direction = get_gaze_direction(left_eye_ratio, right_eye_ratio)

        cv2.putText(frame, gaze_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
