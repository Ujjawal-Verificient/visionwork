import os
import cv2
from ultralytics import YOLO
import numpy as np

# model_to_use = "/home/ajeet/Downloads/v8_nano_production_1.pt"
model_to_use = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain/runs/detect/train8_AfterInternet_e_2_freeze_-1_1e-07/weights/best.pt"
model = YOLO(model_to_use).to("cuda")

base_input_dir = "/home/ajeet/codework/folder/"
output_base_dir = "/home/ajeet/codework/results_on_30videos/pd/finetuned_allsession_scale_1.05_iou0.75_firstther_0.65"
mp_detections_dir = os.path.join(output_base_dir, "mp")

mp_conf_threshold = 0.65
mp_class_to_detect = [0]
iou_threshold = 0.75

os.makedirs(mp_detections_dir, exist_ok=True)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

# def rotate_image(image, angle):
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     return cv2.warpAffine(image, M, (w, h))

def apply_nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, mp_conf_threshold, iou_threshold)
    if isinstance(indices, np.ndarray):  # Handle as a list of indices
        indices = indices.flatten()  # Flatten the array to a list
    elif np.isscalar(indices):  # Handle scalar case
        indices = [indices]
    return [boxes[i] for i in indices]

# def rotate_image(image, angle):
#     center = tuple(np.array(image.shape[1::-1]) / 2)
#     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#     return rotated_image

# def translate_image(image, x_shift, y_shift):
#     h, w = image.shape[:2]
#     translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
#     translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
#     return translated_image

def scale_image(image, scale_factor):
    h, w = image.shape[:2]
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)

    scaled_image = cv2.resize(image, (new_width, new_height))

    # pad_width = (w - new_width) // 2
    # pad_height = (h - new_height) // 2
    # padded_image = cv2.copyMakeBorder(
    #     scaled_image,
    #     pad_height,
    #     h - new_height - pad_height,
    #     pad_width,
    #     w - new_width - pad_width,
    #     cv2.BORDER_CONSTANT,
    #     value=[0, 0, 0],  # Black padding
    # )
    # return padded_image

    return scaled_image

count = 0
detected_before_rotation = 0 
detected_after_rotation = 0 
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
        count += 1
        image = cv2.imread(image_path)
        # if image is None:
        #     print(f"Could not load image: {image_path}")
        #     continue

        
        results_orig = model.predict(
            image_path, conf=mp_conf_threshold, imgsz=640, classes=mp_class_to_detect, 
            # save=True, save_txt=True, show_labels=True, show_conf=True, show_boxes=True
        )
        # boxes_orig = results_orig[0].boxes.xyxy.cpu().numpy() if results_orig[0].boxes else []

        if results_orig[0].boxes is not None:
            boxes_orig = results_orig[0].boxes.xyxy.cpu().numpy()
            scores_orig = results_orig[0].boxes.conf.cpu().numpy()
            boxes_orig = apply_nms(boxes_orig.tolist(), scores_orig.tolist(), 0.4)

            if len(boxes_orig) > 1:
                detected_before_rotation = detected_before_rotation + 1

        # rotated_image = rotate_image(image, 2)
        # rotated_image = translate_image(image, x_shift=5, y_shift=5)

        rotated_image = scale_image(image, 1.05)

        results_rotated = model.predict(
            rotated_image, conf=0.01, imgsz=640, classes=mp_class_to_detect, 
            # save=True, save_txt=True, show_labels=True, show_conf=True, show_boxes=True
        )
        # boxes_rotated = results_rotated[0].boxes.xyxy.cpu().numpy() if results_rotated[0].boxes else []

        if results_rotated[0].boxes is not None:
            boxes_rotated = results_rotated[0].boxes.xyxy.cpu().numpy()
            scores_rotated = results_rotated[0].boxes.conf.cpu().numpy()
            boxes_rotated = apply_nms(boxes_rotated.tolist(), scores_rotated.tolist(), 0.4)

            if len(boxes_rotated) > 1: 
                detected_after_rotation = detected_after_rotation + 1

        true_positives = []
        for box1 in boxes_orig:
            for box2 in boxes_rotated:
                iou = calculate_iou(box1, box2)
                if iou >= iou_threshold:
                    true_positives.append(box1)
                    break

        if len(true_positives) > 1:
            for tp in true_positives:
                x1, y1, x2, y2 = map(int, tp[:4])
                confidence = tp[4] if len(tp) > 4 else 0.0
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"Person {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            output_path = os.path.join(mp_detections_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, image)
            print(f"True positives saved for {image_path}.")

print(f"Model: {model_to_use}")
print(f"Processed {count} images.")
print(detected_before_rotation)
print(f"detected_after_rotation: {detected_after_rotation}")
