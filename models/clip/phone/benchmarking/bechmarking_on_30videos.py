import os
import cv2
from ultralytics import YOLO
import torch
import shutil

# model_to_use = "/home/ajeet/Downloads/v8_nano_production_1.pt"
# model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/train8_AfterInternet_e_2_freeze_-1_1e-07/weights/best.pt"
# model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/best_train/weights/best.pt"
# model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/On_old_bk_train/weights/best.pt"
model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/Oland_and_new_withbk_cts_g5k_133int/weights/best.pt"
# model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/train7/weights/best.pt"
# model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/old_and_new_with_cts_3k_g_2.9val_e_20but21train2/weights/epoch10.pt"
# model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/old_and_new_with_cts_3k_g_2.9val_e_25_train2/weights/best.pt"


model = YOLO(model_to_use, verbose=True).to("cuda")

base_input_dir = "/home/ajeet/codework/datasets/allSessionFrames"
# base_input_dir = "/home/ajeet/codework/datasets/dataset_frames"
output_base_dir = "/home/ajeet/codework/results_on_30videos/ft/Oland_and_new_withbk_cts_g5k_133int"

phone_conf_threshold = 0.50
phone_class_to_detect = [2]

mp_conf_threshold = 0.65
mp_class_to_detect = [0]

phone_detections_dir = os.path.join(output_base_dir, "phone")
mp_detections_dir = os.path.join(output_base_dir, "mp")

count = 0 
for subfolder in os.listdir(base_input_dir):
    subfolder_path = os.path.join(base_input_dir, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    print(f"Processing subfolder: {subfolder}")

    test_image_files = [
        os.path.join(subfolder_path, f)
        for f in os.listdir(subfolder_path)
        if f.lower().endswith(".jpg")
    ]

    for image_path in test_image_files:
        count = count + 1
        results = model.predict(
            image_path,
            conf=phone_conf_threshold,
            imgsz=640,
            classes=phone_class_to_detect,
        )

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                boxes_list = boxes.xyxy.cpu().numpy()
                scores_list = boxes.conf.cpu().numpy()

                indices = cv2.dnn.NMSBoxes(
                    boxes_list.tolist(), scores_list.tolist(), phone_conf_threshold, 0.4
                )

                if len(indices) > 0:
                    for i in indices.flatten():
                        x1, y1, x2, y2 = map(int, boxes_list[i])
                        confidence = scores_list[i]

                        cv2.rectangle(
                            result.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2
                        )
                        cv2.putText(
                            result.orig_img,
                            f"phone {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    image_name = os.path.basename(result.path)
                    detection_path = os.path.join(phone_detections_dir, image_name)
                    cv2.imwrite(detection_path, result.orig_img)
                    print(f"Phone detections saved for {image_name}.")

    for image_path in test_image_files:
        results = model.predict(
            image_path,
            conf=mp_conf_threshold,
            imgsz=640,
            classes=mp_class_to_detect,
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), mp_conf_threshold, 0.4
            )

            person_count = len(indices)
            if person_count > 1:
                for i in indices.flatten():
                    x1, y1, x2, y2 = map(int, boxes[i])
                    confidence = scores[i]

                    cv2.rectangle(
                        result.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        result.orig_img,
                        f"Person {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                image_name = os.path.basename(result.path)
                detection_path = os.path.join(mp_detections_dir, image_name)
                cv2.imwrite(detection_path, result.orig_img)
                print(f"Multiple persons detected for {image_name}.")

print(model_to_use)
print(count)