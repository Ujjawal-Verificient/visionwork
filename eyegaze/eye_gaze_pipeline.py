import os
import cv2
import math
import mediapipe as mp
from iris_folder import filter_fsla
from blinking import BlinkDetector
import sys

retina_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RetinaNet_Face_Verification'))
sys.path.append(retina_path)

from retinanet_test import ajeesing_test_method

import numpy as np

class EyeGazeAnalyzerMP:
    def __init__(self, save_dir=""):
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # os.makedirs(self.save_dir, exist_ok=True)

    def _get_eye_crop(self, landmarks, eye_idx, image, gray, img_path, label):
        h, w = image.shape[:2]
        points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_idx]
        eye_region = np.array(points, np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])
        cropped_eye = eye[min_y:max_y, min_x:max_x]

        parts = img_path.strip().split("/")
        if len(parts) >= 2:
            folder = parts[-2]
            image_name = parts[-1].split(".")[0]
            img_id = f"{folder}_{image_name}"
        else:
            img_id = "unknown"

        eye_path = os.path.join("/home/ajeet/codework/testing_gaze/mask", f"{label}_{img_id}_cropped.jpg")

        if cropped_eye.size > 0:
            cv2.imwrite(eye_path, cropped_eye)

        return cropped_eye

    def _white_pixel_ratio(self, cropped_eye, label):
        if cropped_eye.size == 0:
            return 0, 0

        _, thresh_eye = cv2.threshold(cropped_eye, 70, 255, cv2.THRESH_BINARY)

        parts = img_path.strip().split("/")
        if len(parts) >= 2:
            folder = parts[-2]
            image_name = parts[-1].split(".")[0]
            img_id = f"{folder}_{image_name}"
        else:
            img_id = "unknown"

        mask_path = os.path.join("/home/ajeet/codework/testing_gaze/mask", f"{label}_{img_id}.jpg")
        if thresh_eye.size > 0:
            cv2.imwrite(mask_path, thresh_eye)

        h, w = thresh_eye.shape
        left_white = cv2.countNonZero(thresh_eye[:, :w // 2])
        right_white = cv2.countNonZero(thresh_eye[:, w // 2:])
        return left_white, right_white

    def get_white_pixel_gaze(self, image, landmarks, img_path):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        left_eye_crop = self._get_eye_crop(landmarks, self.LEFT_EYE_IDX, image, gray, img_path, "left")
        right_eye_crop = self._get_eye_crop(landmarks, self.RIGHT_EYE_IDX, image, gray, img_path, "right")

        left_lw, left_rw = self._white_pixel_ratio(left_eye_crop, "left")
        right_lw, right_rw = self._white_pixel_ratio(right_eye_crop, "right")
        print(left_lw, left_rw)
        print(right_lw, right_rw)

        # valid_lw = max(left_lw, right_lw) <= 2 * min(left_lw, right_lw) if min(left_lw, right_lw) > 0 else False
        # valid_rw = max(left_rw, right_rw) <= 2 * min(left_rw, right_rw) if min(left_rw, right_rw) > 0 else False

        # if valid_lw and valid_rw:
        #     total_left_white = left_lw + right_lw
        #     total_right_white = left_rw + right_rw
        # else:
        #     total_left_white = -1
        #     total_right_white = -1

        # total_left_white = left_lw + right_lw
        # total_right_white = left_rw + right_rw

        total_left_white = left_lw 
        total_right_white = left_rw

        return total_left_white, total_right_white

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.blink_detector = BlinkDetector()
        self.white_pixel_detector = EyeGazeAnalyzerMP()

    def euclidean_distance(self, p1, p2, w, h):
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        return math.hypot(x2 - x1, y2 - y1)

    def get_face_bbox_area(self, landmarks, w, h):
        xs = [int(pt.x * w) for pt in landmarks]
        ys = [int(pt.y * h) for pt in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def get_eye_ratio(self, outer, inner, pupil):
        return (pupil.x - outer.x) / (inner.x - outer.x)

    def get_gaze_direction(self, image, landmarks, img_path):
        # h, w = image.shape[:2]
        # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # results = self.face_mesh.process(rgb)


        left_ratio = self.get_eye_ratio(landmarks[33], landmarks[133], landmarks[468])
        right_ratio = self.get_eye_ratio(landmarks[362], landmarks[263], landmarks[473])
        avg_ratio = (left_ratio + right_ratio) / 2.0

        ratio_min = min(left_ratio, right_ratio)
        ratio_max = max(left_ratio, right_ratio)

        total_left_white, total_right_white = self.white_pixel_detector.get_white_pixel_gaze(image, landmarks, img_path)

        white_pixel_ratio = 0
        if total_right_white >= 0:
            white_pixel_ratio = total_left_white/(total_right_white+0.00001)

        pixel_based_gaze = "Looking RIGHT"
        if white_pixel_ratio < 0.5:
            pixel_based_gaze = "Looking RIGHT"
        elif white_pixel_ratio > 1.5:
            pixel_based_gaze = "Looking LEFT"


        ratio_based_gaze = "Looking CENTER"
        if ratio_max <= 1.5 * ratio_min:
            if avg_ratio < 0.38:
                ratio_based_gaze = "Looking RIGHT"
            elif avg_ratio > 0.62:
                ratio_based_gaze = "Looking LEFT"

        final_gaze = "Looking CENTER"
        if ratio_based_gaze == pixel_based_gaze:
            final_gaze = ratio_based_gaze

        debug_info = f"LRatio: {left_ratio:.2f}, RRatio: {right_ratio:.2f}, Avg: {avg_ratio:.2f}, " \
                        f"WhiteL: {total_left_white}, WhiteR: {total_right_white}, Gaze: {final_gaze}"
        print(debug_info)

        return final_gaze, left_ratio, right_ratio, avg_ratio, total_left_white, total_right_white


if __name__ == '__main__':
    # parent_folder = "/home/ajeet/codework/datasets/dataset_frames"
    parent_folder = "/home/ajeet/Downloads/ragini_FSLA_videos/frames"
    output_base = "/home/ajeet/codework/testing_gaze"

    gaze_detector = GazeDetector()
    blink_detector = gaze_detector.blink_detector

    white_pixel_detector = EyeGazeAnalyzerMP()

    output_folders = {
        "Looking LEFT": os.path.join(output_base, "left"),
        "Looking RIGHT": os.path.join(output_base, "right"),
        "Looking CENTER": os.path.join(output_base, "center"),
        "Face too far": os.path.join(output_base, "face_too_far"),
        "Blinking": os.path.join(output_base, "blinking"),
        "Head Turned": os.path.join(output_base, "head_turned"),
        "Face not detected": os.path.join(output_base, "face_not_detected"),
        "Landmark error": os.path.join(output_base, "landmark_error"),
        "Not_in_1.5": os.path.join(output_base, "Not_in_1.5")
    }

    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    count = 1
    for subfolder in os.listdir(parent_folder):
        input_folder = os.path.join(parent_folder, subfolder)
        if not os.path.isdir(input_folder):
            continue

        print(f"\nProcessing folder: {input_folder}")
        print(f"\ncount: {count}")
        count = count + 1

        for img_file in os.listdir(input_folder):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(input_folder, img_file)
            save_actual_image = img_path

            actual_image_frame= cv2.imread(save_actual_image)

            # ajeesing_test_method(img_path)

            # result_img_path = "/home/ajeet/codework/mediapipeDemos/image/result12.jpg"
            # img_path = result_img_path
            # if not os.path.exists(result_img_path):
            #     continue

            img_path = img_path

            if  not filter_fsla(img_path):
                dst = os.path.join(output_folders["Head Turned"], f"{subfolder}_{img_file}")
                cv2.imwrite(dst, actual_image_frame)
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = gaze_detector.face_mesh.process(rgb_frame)

            gaze = "Face not detected"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                landmark_indices = [33, 133, 468, 362, 263, 473]
                landmark_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 0, 255)]

                for idx, color in zip(landmark_indices, landmark_colors):
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 1, color, -1) 


                blink_result = blink_detector.is_blinking(landmarks, w, h)
                blinking, avg_ear, (left_ear, right_ear) = blink_result

                if blinking is not None:
                    cv2.putText(frame, f"Blinking EAR: {avg_ear:.2f}", (10, h - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                if blinking is not None and blinking:
                    output_path = os.path.join(output_folders["Blinking"], f"{subfolder}_{img_file}")
                    cv2.imwrite(output_path, frame)
                    continue

                
                final_gaze, left_ratio, right_ratio, avg_ratio, total_left_white, total_right_white = gaze_detector.get_gaze_direction(frame, landmarks, img_path)
                gaze = final_gaze

                cv2.putText(frame, f"Left: {left_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Right: {right_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Avg: {avg_ratio:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.putText(frame, f"LW: {total_left_white}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"RW: {total_right_white}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"{final_gaze}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            save_path = output_folders.get(gaze, output_folders[gaze])
            output_path = os.path.join(save_path, f"{subfolder}_{img_file}")
            cv2.imwrite(output_path, frame)


# if __name__ == '__main__':
#     # parent_folder = "/home/ajeet/codework/datasets/dataset_frames"
#     parent_folder = "/home/ajeet/Downloads/FSLA_TP_FP_VIDEOS/frames"
#     output_base = "/home/ajeet/codework/testing_gaze"

#     gaze_detector = GazeDetector()
#     blink_detector = gaze_detector.blink_detector

#     white_pixel_detector = EyeGazeAnalyzerMP()

#     output_folders = {
#         "Looking LEFT": os.path.join(output_base, "left"),
#         "Looking RIGHT": os.path.join(output_base, "right"),
#         "Looking CENTER": os.path.join(output_base, "center"),
#         "Face too far": os.path.join(output_base, "face_too_far"),
#         "Blinking": os.path.join(output_base, "blinking"),
#         "Head Turned": os.path.join(output_base, "head_turned"),
#         "Face not detected": os.path.join(output_base, "face_not_detected"),
#         "Landmark error": os.path.join(output_base, "landmark_error"),
#         "Not_in_1.5": os.path.join(output_base, "Not_in_1.5")
#     }

#     for folder in output_folders.values():
#         os.makedirs(folder, exist_ok=True)

#     count = 1
#     cap = cv2.VideoCapture(0)  # 0 is the default laptop webcam

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         exit()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = gaze_detector.face_mesh.process(rgb_frame)

#         img_path = f"/home/ajeet/codework/testing_gaze/images/debug_frame_{count}.jpg"
#         cv2.imwrite(img_path, frame)

#         gaze = "Face not detected"

#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark

#             # Draw specific landmarks on the actual image
#             landmark_indices = [33, 133, 468, 362, 263, 473]
#             landmark_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 0, 255)]

#             for idx, color in zip(landmark_indices, landmark_colors):
#                 x = int(landmarks[idx].x * w)
#                 y = int(landmarks[idx].y * h)
#                 cv2.circle(frame, (x, y), 4, color, -1)  # radius 4, filled circle


#             blink_result = blink_detector.is_blinking(landmarks, w, h)
#             blinking, avg_ear, (left_ear, right_ear) = blink_result

#             if blinking is not None:
#                 cv2.putText(frame, f"Blinking EAR: {avg_ear:.2f}", (10, h - 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
#             if blinking is not None and blinking:
#                 output_path = os.path.join(output_folders["Blinking"], f"{count}.jpg")
#                 cv2.imwrite(output_path, frame)
#                 continue

            
#             final_gaze, left_ratio, right_ratio, avg_ratio, total_left_white, total_right_white = gaze_detector.get_gaze_direction(frame, landmarks, img_path)
#             gaze = final_gaze

#                     # Display non-overlapping text on image
#             cv2.putText(frame, f"Left Ratio: {left_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#             cv2.putText(frame, f"Right Ratio: {right_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#             cv2.putText(frame, f"Avg Ratio: {avg_ratio:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             cv2.putText(frame, f"Left White: {total_left_white}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Right White: {total_right_white}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Gaze: {final_gaze}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#         save_path = output_folders.get(gaze, output_folders[gaze])
#         output_path = os.path.join(save_path, f"{count}.jpg")
#         cv2.imwrite(output_path, frame)
#         count = count + 1

#         cv2.imshow("Live Gaze Detection", frame)

#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

